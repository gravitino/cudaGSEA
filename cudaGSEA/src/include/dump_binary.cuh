#ifndef CUDA_GSEA_DUMP_BINARY
#define CUDA_GSEA_DUMP_BINARY

#include <string>
#include <fstream>

#include "cuda_helpers.cuh"
#include "configuration.cuh"

template <class index_t, class value_t>
void dump_binary(const value_t * data, const index_t L, std::string filename) {

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_dump_to_disk)
    #endif

    #ifdef CUDA_GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Dumping " << (sizeof(value_t)*L) << " many bytes to "
              << filename << "." << std::endl;
    #endif

    // write out file for visual inspection
    std::ofstream ofile(filename.c_str(), std::ios::binary);
    ofile.write((char*) data, sizeof(value_t)*L);
    ofile.close();

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_dump_to_disk)
    #endif
}

#endif
