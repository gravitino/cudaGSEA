#ifndef CUDA_GSEA_CREATE_BITMAPS_CUH
#define CUDA_GSEA_CREATE_BITMAPS_CUH

#include <unordered_set>
#include <vector>
#include <string>

#include <assert.h>

#include "cuda_helpers.cuh"
#include "configuration.cuh"

template <class bitmp_t, class index_t> __host__
void create_bitmaps(const std::vector<std::string>& gsymb,
                    const std::vector<std::vector<std::string>>& pathw,
                    std::vector<bitmp_t>& opath,
                    index_t lower,
                    index_t width) {

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_create_bitmaps)
    #endif

    assert(width <= sizeof(bitmp_t)*8);

    // prepare the original paths vector
    opath.resize(gsymb.size());

    // construct sets of pathways
    std::vector<std::unordered_set<std::string>> sets;
    for (index_t id = lower; id < lower+width; id++) {
        std::unordered_set<std::string> set;
        for (const auto& symbol : pathw[id])
            set.insert(symbol);
        sets.push_back(std::move(set));
    }

    // compute bitmaps
    size_t gene = 0;
    for (const auto& symbol : gsymb) {

        // std::cout << "SYMBOLKEY " << symbol << std::endl;

        bitmp_t bitmap; bitmap.zero();

        size_t path = 0;
        for (const auto& set : sets) {

            if (set.find(symbol) != set.end()) {
                bitmap.setbit(path);
                
                //if (path == 0)
                //    std::cout << gene << ",";

            }
            path++;
        }

        opath[gene++] = bitmap;
    }

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_create_bitmaps)
    #endif
}

#endif
