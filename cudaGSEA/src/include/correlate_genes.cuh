#ifndef CUDA_GSEA_CORRELATE_GENES
#define CUDA_GSEA_CORRELATE_GENES

#include <omp.h>                   // openMP
#include <assert.h>                // checks
#include "functors.cuh"            // functors
#include "error_codes.cuh"         // error codes
#include "cuda_helpers.cuh"        // timings
#include "configuration.cuh"       // configuration
#include "rngpu/rngpu.hpp"         // lightweight curand alternative

//////////////////////////////////////////////////////////////////////////////
// correlate low-level primitives for CPU and GPU
//////////////////////////////////////////////////////////////////////////////

template <
    class label_t,
    class index_t,
    class value_t,
    class funct_t,
    unsigned int seed=CUDA_GSEA_PERMUTATION_SEED> __global__
void correlate_gpu(
    value_t * table,       // expression data table (constin)
    label_t * labels,      // class labels for the two phenotypes
    value_t * correl,      // correlation of genes (output)
    index_t num_genes,     // number of genes
    index_t num_type_A,    // number of patients phenotype A
    index_t num_type_B,    // number of patients phenotype B
    index_t num_perms,     // number of permutations
    funct_t accum,         // accumulator functor
    index_t shift=0) {     // permutation shift

    // indices and helper variables
    const index_t lane = num_type_A + num_type_B;
    const index_t blid = blockIdx.x;
    const index_t thid = threadIdx.x;

    // shared memory for labels for the later permutation
    extern __shared__ label_t sigma[];

    // for each permutation
    for (index_t perm = blid; perm < num_perms; perm += gridDim.x) {

        // create default permutation in shared memory
        for (index_t id = thid; id < lane; id += blockDim.x)
             sigma[id] = labels[id];
        __syncthreads();

        // the first thread shuffles the permutation except permutation 0
        if ((thid == 0) && (perm+shift)) {

            // Fisher-Yates shuffle (carry and add fast kiss rng)
            auto state=get_initial_fast_kiss_state32(perm+shift+seed);
            fisher_yates_shuffle(fast_kiss32, &state, sigma, lane);
        }
        __syncthreads();

        // for each gene accumulate contribution over all patients
        for (index_t gene = thid; gene < num_genes; gene += blockDim.x)
            correl[perm*num_genes+gene] = accum(sigma,
                                                table,
                                                lane,
                                                gene,
                                                num_genes,
                                                num_type_A,
                                                num_type_B);
    }
}

template <
    class label_t,
    class index_t,
    class value_t,
    class funct_t,
    unsigned int seed=CUDA_GSEA_PERMUTATION_SEED> __global__
void correlate_cpu(
    value_t * table,       // expression data table (constin)
    label_t * labels,      // class labels for the two phenotypes
    value_t * correl,      // correlation of genes (output)
    index_t num_genes,     // number of genes
    index_t num_type_A,    // number of patients phenotype A
    index_t num_type_B,    // number of patients phenotype B
    index_t num_perms,     // number of permutations
    funct_t accum,         // accumulator functor
    index_t shift=0) {     // permutation shift

    // indices and helper variables
    const index_t lane = num_type_A + num_type_B;

    // for each permutation
    # pragma omp parallel for
    for (index_t perm = 0; perm < num_perms; perm++) {

        // copy label vector
        std::vector<label_t> sigma(lane);
        for (index_t patient = 0; patient < lane; patient++)
            sigma[patient] = labels[patient];

        // create permutation
        if (perm+shift > 0) {
            auto state=get_initial_fast_kiss_state32(perm+shift+seed);
            fisher_yates_shuffle(fast_kiss32, &state, sigma.data(), lane);
        }

        // for each gene accumulate contribution over all patients
        for (index_t gene = 0; gene < num_genes; gene++)
            correl[perm*num_genes+gene] = accum(sigma.data(),
                                                table,
                                                lane,
                                                gene,
                                                num_genes,
                                                num_type_A,
                                                num_type_B);
    }
}

//////////////////////////////////////////////////////////////////////////////
// correlate high-level primitives for CPU and GPU
//////////////////////////////////////////////////////////////////////////////

template <
    class value_t,
    class label_t,
    class index_t,
    bool biased=true,     // use biased standard deviation if applicable
    bool fixlow=true,     // adjust low standard deviation if applicable
    bool transposed=true> // do not change this unless you know better
void correllate_genes_gpu(
    value_t * Exprs,         // expression data  (constin)
    label_t * Labels,        // phenotype labels (constin)
    value_t * Correl,        // correlation of genes (output)
    index_t num_genes,       // number of genes
    index_t num_type_A,      // number patients phenotype A
    index_t num_type_B,      // number patients phenotype B
    index_t num_perms,       // number of permutations
    std::string metric,      // specify the used metric
    index_t shift=0,         // shift in permutations
    cudaStream_t stream=0) { // specified CUDA stream

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(device_correlate_genes)
    #endif

    #ifdef CUDA_GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Correlating " << num_genes << " unique gene symbols "
              << "for " << num_type_A << " patients" << std::endl
              << "STATUS: with phenotype 0 and " << num_type_B
              << " patients with phenotype 1" << std::endl
              << "STATUS: over " << num_perms
              << " permutations (shift=" << shift << ") " << std::endl
              << "STATUS: using " << metric << " as ranking metric on the GPU."
              << std::endl;
    #endif

    // consistency checks
    assert(num_type_A > 0);
    assert(num_type_B > 0);
    assert(num_genes > 0);
    assert(num_perms > 0);

    // all kernels demand some shared memory
    const auto shared_mem_size = (num_type_A+num_type_B)*sizeof(label_t);
    const auto num_threads = 256;

    // choose metric
    if (metric == "naive_diff_of_classes") {
        auto combiner = combine_difference();
        auto functor  = accumulate_naive_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "naive_ratio_of_classes") {
        auto combiner = combine_quotient();
        auto functor  = accumulate_naive_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "naive_log2_ratio_of_classes") {
        auto combiner = combine_log2_quotient();
        auto functor  = accumulate_naive_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "stable_diff_of_classes") {
        auto combiner = combine_difference();
        auto functor  = accumulate_kahan_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "stable_ratio_of_classes") {
        auto combiner = combine_quotient();
        auto functor  = accumulate_kahan_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "stable_log2_ratio_of_classes") {
        auto combiner = combine_log2_quotient();
        auto functor  = accumulate_kahan_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "onepass_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_onepass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "onepass_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_onepass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "twopass_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_twopass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "twopass_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_twopass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "stable_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_knuth_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "stable_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_knuth_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "overkill_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_overkill_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else if (metric == "overkill_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_overkill_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_gpu<label_t>
                     <<<num_perms, num_threads, shared_mem_size, stream>>>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             CUERR

    } else {
        std::cout << "ERROR: unknown metric, exiting." << std::endl;
        exit(CUDA_GSEA_NO_KNOWN_METRIC_SPECIFIED_ERROR);
    }

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(device_correlate_genes)
    #endif
}

#endif
