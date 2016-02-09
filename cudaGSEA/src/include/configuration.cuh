#ifndef CUDA_GSEA_CONFIGURATION_CUH
#define CUDA_GSEA_CONFIGURATION_CUH

// documentation level settings
//#define CUDA_GSEA_PRINT_VERBOSE
//#define CUDA_GSEA_PRINT_TIMINGS
#define CUDA_GSEA_PRINT_INFO
#define CUDA_GSEA_PRINT_WARNINGS

// RAM scheduler settings (should be smaller 1 and positive)
#define CUDA_GSEA_FREE_RAM_SECURITY_FACTOR (0.97)

// seed for sampling permutations
#define CUDA_GSEA_PERMUTATION_SEED (42)

#endif
