#ifndef CUDA_GSEA_BATCHED_SORT
#define CUDA_GSEA_BATCHED_SORT

// CUB dynamic parallelism
#ifndef CUB_CDP
#define CUB_CDP
#endif

#include <assert.h>
#include "cub/cub.cuh"
#include "transpose.cuh"
#include "cuda_helpers.cuh"
#include "configuration.cuh"

///////////////////////////////////////////////////////////////////////////////
// Batched sort using device-wide CUB primitives
///////////////////////////////////////////////////////////////////////////////

template <class key_t, class val_t, class index_t, 
          bool transposed=false, bool ascending=false, index_t num_streams=32>
void pathway_sort(key_t * Keys,           // keys of all batches   (inplace)
                  val_t * Source,         // values for one batch  (constin)
                  val_t * Target,         // values of all batches (inplace)
                  index_t batch_size,     // number of entries in a batch
                  index_t num_batches) {  // number of batches

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(device_sort_correlated_genes)
    #endif

    #ifdef CUDA_GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Sort " << num_batches << " correlation arrays "
              << std::endl << "STATUS: each of size " << batch_size 
              << " on the GPU." << std::endl;
    #endif

    // CUB storages
    size_t temp_storage_bytes = 0;
    void *d_temp_storage[num_streams];

    // stream management
    cudaStream_t streams[num_streams];
    key_t * temp_keys = nullptr;

    // dry run for one batch
    if (ascending)
        cub::DeviceRadixSort::SortPairs(nullptr, 
                                        temp_storage_bytes,
                                        Keys, 
                                        temp_keys, 
                                        Source, 
                                        Target, 
                                        batch_size);
    else
        cub::DeviceRadixSort::SortPairsDescending(nullptr, 
                                                  temp_storage_bytes,
                                                  Keys, 
                                                  temp_keys, 
                                                  Source, 
                                                  Target, 
                                                  batch_size);

    // create streams and temp storage
    cudaMalloc(&temp_keys, sizeof(key_t)*batch_size*num_streams);
    for (index_t stream = 0; stream < num_streams; stream++) {
        cudaStreamCreate(streams+stream);
        cudaMalloc(&d_temp_storage[stream], temp_storage_bytes);
    }

    // for each batch pair sort
    for (index_t batch = 0; batch < num_batches; batch++) {

        // determine current stream
        const index_t stream = batch % num_streams;

        // sort
        if (ascending)
            cub::DeviceRadixSort::SortPairs
                                  (d_temp_storage[stream], 
                                   temp_storage_bytes,
                                   Keys+batch*batch_size, 
                                   temp_keys+stream*batch_size, 
                                   Source, 
                                   Target+batch*batch_size, 
                                   batch_size,
                                   0, sizeof(key_t)*8, 
                                   streams[stream]);
        else
            cub::DeviceRadixSort::SortPairsDescending
                                  (d_temp_storage[stream], 
                                   temp_storage_bytes,
                                   Keys+batch*batch_size, 
                                   temp_keys+stream*batch_size, 
                                   Source, 
                                   Target+batch*batch_size, 
                                   batch_size,
                                   0, sizeof(key_t)*8, 
                                   streams[stream]);

        // if num_streams many streams have been seen copy result back
        if (stream+1 == num_streams) {
            const index_t shift = (batch+1-num_streams)*batch_size;
            cudaMemcpy(Keys+shift, 
                       temp_keys, 
                       sizeof(key_t)*batch_size*num_streams, 
                       cudaMemcpyDeviceToDevice);
        }
    }

    // copy the remaining parts
    const index_t remaining = num_batches%num_streams;
    if (remaining) {
        const index_t shift = (num_batches-remaining)*batch_size;
        cudaMemcpy(Keys+shift, 
                   temp_keys, 
                   sizeof(key_t)*batch_size*remaining, 
                   cudaMemcpyDeviceToDevice);
    }

    // get rid of the memory
    cudaFree(temp_keys);
    for (index_t stream = 0; stream < num_streams; stream++) {
        cudaStreamSynchronize(streams[stream]);
        cudaStreamDestroy(streams[stream]);
        cudaFree(d_temp_storage[stream]);
    }
    
    cudaDeviceSynchronize();

    if (transposed) {
        transpose(Target, num_batches, batch_size);
        transpose(Keys, num_batches, batch_size);
    }

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(device_sort_correlated_genes)
    #endif

}

#endif
