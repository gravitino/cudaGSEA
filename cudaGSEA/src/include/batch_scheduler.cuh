#ifndef CUDA_GSEA_BATCH_SCHEDULER_CUH
#define CUDA_GSEA_BATCH_SCHEDULER_CUH

#include <iostream>
#include "cuda_helpers.cuh"
#include "configuration.cuh"

template <
    class exprs_t,
    class bitmp_t,
    class enrch_t,
    class index_t>
void schedule_computation_gpu(
    index_t  num_genes,
    index_t  num_perms,
    index_t  num_paths,
    index_t& batch_size_perms,
    index_t& num_batches_perms,
    index_t& batch_size_paths,
    index_t& num_batches_paths,
    index_t& num_free_bytes) {

    // estimate number of paths in one batch and number of batches
    batch_size_paths = cumin(8*sizeof(bitmp_t), num_paths);
    num_batches_paths = (num_paths+batch_size_paths-1)/batch_size_paths;

    // estimate number of permutations in one batch and number of batches
    num_free_bytes = freeRAM()*CUDA_GSEA_FREE_RAM_SECURITY_FACTOR;
    index_t pi = (num_free_bytes-num_genes*sizeof(bitmp_t)) /
                 (num_genes*(sizeof(exprs_t)+sizeof(unsigned int))
                 +8*sizeof(enrch_t)*sizeof(bitmp_t));
    batch_size_perms = cumin(num_perms, pi);
    num_batches_perms = (num_perms+batch_size_perms-1)/batch_size_perms;

    #ifdef CUDA_GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Could fit up to " << pi
              << " permutations at once in RAM of the GPU." << std::endl;
    #endif
}

template <
    class exprs_t,
    class bitmp_t,
    class enrch_t,
    class index_t>
void schedule_computation_cpu(
    index_t  num_genes,
    index_t  num_perms,
    index_t  num_paths,
    index_t& batch_size_perms,
    index_t& num_batches_perms,
    index_t& batch_size_paths,
    index_t& num_batches_paths,
    index_t& num_free_bytes) {

    // estimate number of paths in one batch and number of batches
    batch_size_paths = cumin(8*sizeof(bitmp_t), num_paths);
    num_batches_paths = (num_paths+batch_size_paths-1)/batch_size_paths;

    // estimate number of permutations in one batch and number of batches
    num_free_bytes = CUDA_GSEA_HOST_FREE_RAM*CUDA_GSEA_FREE_RAM_SECURITY_FACTOR;
    index_t pi = (num_free_bytes-num_genes*sizeof(bitmp_t)) /
                 (num_genes*(sizeof(exprs_t)+sizeof(unsigned int))
                 +8*sizeof(enrch_t)*sizeof(bitmp_t));
    batch_size_perms = cumin(num_perms, pi);
    num_batches_perms = (num_perms+batch_size_perms-1)/batch_size_perms;

    #ifdef CUDA_GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Could fit up to " << pi
              << " permutations at once in RAM of the CPU." << std::endl;
    #endif
}


#endif
