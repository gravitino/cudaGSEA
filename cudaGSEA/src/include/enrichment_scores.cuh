#ifndef CUDA_GSEA_ENRICHMENT_SCORES_CUH
#define CUDA_GSEA_ENRICHMENT_SCORES_CUH

#include "math_helpers.cuh"
#include "cuda_helpers.cuh"
#include "configuration.cuh"

#ifdef CUDA_GSEA_OPENMP_ENABLED
#inlcude <omp.h>
#endif

template<
    class exprs_t,
    class posit_t,
    class bitmp_t,
    class enrch_t,
    class index_t,
    bool transposed,
    bool kahan> __global__
void score_kernel(
    exprs_t * Correl,    // num_perms x num_genes (constin)
    posit_t * Index,     // num_perms x num_genes (constin)
    bitmp_t * OPath,     // num_genes (constin)
    enrch_t * Score,     // num_paths x num_perms (out)
    index_t num_genes,   // number of genes
    index_t num_perms,   // number of permutations
    index_t num_paths) { // number of pathways

    const index_t blid = blockIdx.x;
    const index_t thid = threadIdx.x;

    for (index_t perm = blid; perm < num_perms; perm += gridDim.x) {
        for (index_t path = thid; path < num_paths; path += blockDim.x) {

           enrch_t Bsf = 0;
           enrch_t Res = 0;
           enrch_t Kah = 0;
           enrch_t N_R = 0;
           index_t N_H = 0;

           // first pass: calculate N_R and N_H
           for (index_t gene = 0; gene < num_genes; gene++) {
                index_t index = transposed ? gene*num_perms+perm :
                                             perm*num_genes+gene ;

                const exprs_t value = Correl[index];
                const exprs_t sigma = OPath[Index[index]].getbit(path);
                const enrch_t input = cuabs(value);

                if (kahan) {
                    const enrch_t alpha = sigma ? input - Kah : 0;
                    const enrch_t futre = sigma ? alpha + N_R : N_R;
                    Kah = sigma ? (futre-N_R)-alpha : Kah;
                    N_R = sigma ? futre : N_R;
                } else {
                    N_R = sigma ? N_R + input : N_R;
                }

                N_H += sigma;
           }

           // second pass: calculate ES
           Kah = 0;
           const enrch_t decay = -1.0/(num_genes-N_H);
           for (index_t gene = 0; gene < num_genes; gene++) {
                index_t index = transposed ? gene*num_perms+perm :
                                             perm*num_genes+gene ;

                const exprs_t value = Correl[index];
                const exprs_t sigma = OPath[Index[index]].getbit(path);
                const enrch_t input = sigma ? cuabs(value)/N_R : decay;

                if (kahan) {
                    const enrch_t alpha = input - Kah;
                    const enrch_t futre = alpha + Res;
                    Kah = (futre-Res)-alpha;
                    Res = futre;
                } else {
                    Res += input;
                }

                Bsf  = cuabs(Res) > cuabs(Bsf) ? Res : Bsf;
           }

           // store best enrichment per path
           Score[path*num_perms+perm] = Bsf;
        }
    }
}

template<
    class exprs_t,
    class posit_t,
    class bitmp_t,
    class enrch_t,
    class index_t,
    bool transposed,
    uint batch_size,
    bool kahan> __global__
void score_kernel_shared(
    exprs_t * Correl,    // num_perms x num_genes (constin)
    posit_t * Index,     // num_perms x num_genes (constin)
    bitmp_t * OPath,     // num_genes (constin)
    enrch_t * Score,     // num_paths x num_perms (out)
    index_t num_genes,   // number of genes
    index_t num_perms,   // number of permutations
    index_t num_paths) { // number of pathways

    const index_t blid = blockIdx.x;
    const index_t thid = threadIdx.x;

    extern __shared__ char cache [];
    bitmp_t * paths_cache = (bitmp_t *) cache;
    exprs_t * corrl_cache = (exprs_t *) (paths_cache+batch_size);


    for (index_t perm = blid; perm < num_perms; perm += gridDim.x) {
        for (index_t path = thid; path < sizeof(bitmp_t)*8; path += blockDim.x){

           enrch_t Bsf = 0;
           enrch_t Res = 0;
           enrch_t Kah = 0;
           enrch_t N_R = 0;
           index_t N_H = 0;

           index_t num_batches = (num_genes+batch_size-1)/batch_size;

           // first pass: calculate N_R and N_H
           for (index_t batch = 0; batch < num_batches; batch++) {
                index_t lower = batch*batch_size;
                index_t upper = lower+batch_size;
                upper = upper > num_genes ? num_genes : upper;

                for (index_t id = thid; id < upper-lower; id += blockDim.x) {
                    index_t index = transposed ?
                                    (lower+id)*num_perms+perm :
                                    perm*num_genes+lower+id;
                    corrl_cache[id] = Correl[index];
                    paths_cache[id] = OPath[Index[index]];
                }
                __syncthreads();

                for (index_t id = 0; id < upper-lower; id++) {
                    const exprs_t value = corrl_cache[id];
                    const exprs_t sigma = paths_cache[id].getbit(path);
                    const enrch_t input = cuabs(value);

                    if (kahan) {
                        const enrch_t alpha = sigma ? input - Kah : 0;
                        const enrch_t futre = sigma ? alpha + N_R : N_R;
                        Kah = sigma ? (futre-N_R)-alpha : Kah;
                        N_R = sigma ? futre : N_R;
                    } else {
                        N_R = sigma ? N_R + input : N_R;
                    }

                    N_H += sigma;
                }
                __syncthreads();
           }

           // second pass: calculate ES
           Kah = 0;
           for (index_t batch = 0; batch < num_batches; batch++) {
                index_t lower = batch*batch_size;
                index_t upper = lower+batch_size;
                upper = upper > num_genes ? num_genes : upper;

                for (index_t id = thid; id < upper-lower; id += blockDim.x) {
                    index_t index = transposed ?
                                    (lower+id)*num_perms+perm :
                                    perm*num_genes+lower+id;
                    corrl_cache[id] = Correl[index];
                    paths_cache[id] = OPath[Index[index]];
                }
                __syncthreads();

                const enrch_t decay = -1.0/(num_genes-N_H);
                for (index_t id = 0; id < upper-lower; id++) {

                    const exprs_t value = corrl_cache[id];
                    const exprs_t sigma = paths_cache[id].getbit(path);
                    const enrch_t input = sigma ? cuabs(value)/N_R : decay;

                   if (kahan) {
                        const enrch_t alpha = input - Kah;
                        const enrch_t futre = alpha + Res;
                        Kah = (futre-Res)-alpha;
                        Res = futre;
                    } else {
                        Res += input;
                    }

                    Bsf = cuabs(Res) > cuabs(Bsf) ? Res : Bsf;
                }

                __syncthreads();

           }

           // store best enrichment per path
           if(path < num_paths)
               Score[path*num_perms+perm] = Bsf;
        }
    }
}

template<
    class exprs_t,
    class posit_t,
    class bitmp_t,
    class enrch_t,
    class index_t,
    bool transposed=false,
    bool shared_memory=true,
    bool kahan=true,
    uint batch=64>
void compute_scores_gpu(
    exprs_t * Correl,    // num_perms x num_genes (constin)
    posit_t * Index,     // num_perms x num_genes (constin)
    bitmp_t * OPath,     // num_genes (constin)
    enrch_t * Score,     // num_paths x num_perms (out)
    index_t num_genes,   // number of genes
    index_t num_perms,   // number of permutations
    index_t num_paths) { // number of pathways

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(device_compute_enrichment_scores)
    #endif

    #ifdef CUDA_GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Computing enrichment scores for " << num_perms
              << " permutations" << std::endl
              << "STATUS: and " << num_paths << " gene sets each over "
              << num_genes << " unique gene symbols." << std::endl;
    #endif

    cudaMemset(Score, 0, sizeof(enrch_t)*num_paths*num_perms);            CUERR

    if (shared_memory) {
        index_t shared_mem_size = batch*(sizeof(exprs_t)+sizeof(bitmp_t));
        score_kernel_shared
        <exprs_t, posit_t, bitmp_t, enrch_t, index_t, transposed, batch, kahan>
        <<<num_perms, sizeof(bitmp_t)*8, shared_mem_size>>>
        (Correl, Index, OPath, Score, num_genes, num_perms, num_paths);   CUERR
    } else {
        score_kernel
        <exprs_t, posit_t, bitmp_t, enrch_t, index_t, transposed, kahan>
        <<<num_perms, sizeof(bitmp_t)*8>>>
        (Correl, Index, OPath, Score, num_genes, num_perms, num_paths);   CUERR
    }

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(device_compute_enrichment_scores)
    #endif
}


template<
    class exprs_t,
    class posit_t,
    class bitmp_t,
    class enrch_t,
    class index_t,
    bool transposed=false,
    bool shared_memory=true,
    bool kahan=true,
    uint batch=64>
void compute_scores_cpu(
    exprs_t * Correl,    // num_perms x num_genes (constin)
    posit_t * Index,     // num_perms x num_genes (constin)
    bitmp_t * OPath,     // num_genes (constin)
    enrch_t * Score,     // num_paths x num_perms (out)
    index_t num_genes,   // number of genes
    index_t num_perms,   // number of permutations
    index_t num_paths) { // number of pathways

    #ifdef CUDA_GSEA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (index_t perm; perm < num_perms; perm++) {
        for (index_t path = 0; path < num_paths; path++) {

            enrch_t Bsf = 0;
            enrch_t Res = 0;
            enrch_t Kah = 0;
            enrch_t N_R = 0;
            index_t N_H = 0;

            // first pass: calculate N_R and N_H
           for (index_t gene = 0; gene < num_genes; gene++) {
                index_t index = transposed ? gene*num_perms+perm :
                                             perm*num_genes+gene ;

                const exprs_t value = Correl[index];
                const exprs_t sigma = OPath[Index[index]].getbit(path);
                const enrch_t input = cuabs(value);

                if (kahan) {
                    const enrch_t alpha = sigma ? input - Kah : 0;
                    const enrch_t futre = sigma ? alpha + N_R : N_R;
                    Kah = sigma ? (futre-N_R)-alpha : Kah;
                    N_R = sigma ? futre : N_R;
                } else {
                    N_R = sigma ? N_R + input : N_R;
                }

                N_H += sigma;
           }

           // second pass: calculate ES
           Kah = 0;
           const enrch_t decay = -1.0/(num_genes-N_H);
           for (index_t gene = 0; gene < num_genes; gene++) {
                index_t index = transposed ? gene*num_perms+perm :
                                             perm*num_genes+gene ;

                const exprs_t value = Correl[index];
                const exprs_t sigma = OPath[Index[index]].getbit(path);
                const enrch_t input = sigma ? cuabs(value)/N_R : decay;

                if (kahan) {
                    const enrch_t alpha = input - Kah;
                    const enrch_t futre = alpha + Res;
                    Kah = (futre-Res)-alpha;
                    Res = futre;
                } else {
                    Res += input;
                }

                Bsf  = cuabs(Res) > cuabs(Bsf) ? Res : Bsf;
           }

           // store best enrichment per path
           Score[path*num_perms+perm] = Bsf;
    	}
    }
}
#endif
