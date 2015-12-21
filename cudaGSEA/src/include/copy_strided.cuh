#ifndef CUDA_GSEA_COPY_STRIDED_CUH
#define CUDA_GSEA_COPY_STRIDED_CUH

#include <algorithm>

#include "cuda_helpers.cuh"
#include "configuration.cuh"

template <class enrch_t, class index_t>
void copy_strided(enrch_t * source,
                  enrch_t * target,
                  index_t num_paths,
                  index_t num_perms,
                  index_t lower_pa,
                  index_t upper_pa,
                  index_t lower_pi,
                  index_t upper_pi) {

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_copy_strided_scores);
    #endif

    for (index_t local_path = 0; local_path < upper_pa-lower_pa; local_path++) {
        const index_t global_offset = (lower_pa+local_path)*num_perms+lower_pi;
        const index_t local_offset = local_path*(upper_pi-lower_pi); 

        std::copy(source+local_offset, 
                  source+local_offset+(upper_pi-lower_pi), 
                  target+global_offset);
    }

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_copy_strided_scores);
    #endif

}

#endif
