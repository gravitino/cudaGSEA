#ifndef CUDA_GSEA_BATCHED_SORT
#define CUDA_GSEA_BATCHED_SORT

// #define STANDALONE_UNIT_TEST
#ifdef STANDALONE_UNIT_TEST
#include "cub/cub.cuh"
#include <iostream>
#include <algorithm>

// delegate transpose to the void (do not try to understand this!)
#define transpose(a,b,c)

// check result if sorted
#define CHECK

// visual inspection
// #define OUTPUT

// error makro
#define CUERR {                                                              \
    cudaError_t err;                                                         \
    if ((err = cudaGetLastError()) != cudaSuccess) {                         \
       std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "       \
                 << __FILE__ << ", line " << __LINE__ << std::endl;          \
       exit(1);                                                              \
    }                                                                        \
}

// convenient timers
#define TIMERSTART(label)                                                    \
        cudaEvent_t start##label, stop##label;                               \
        float time##label;                                                   \
        cudaEventCreate(&start##label);                                      \
        cudaEventCreate(&stop##label);                                       \
        cudaEventRecord(start##label, 0);

#define TIMERSTOP(label)                                                     \
        cudaEventRecord(stop##label, 0);                                     \
        cudaEventSynchronize(stop##label);                                   \
        cudaEventElapsedTime(&time##label, start##label, stop##label);       \
        std::cout << time##label << " ms (" << #label << ")" << std::endl;
#else
//#include <assert.h>
#include "cub/cub.cuh"
#include "transpose.cuh"
#include "cuda_helpers.cuh"
#include "configuration.cuh"
#include "cub/cub.cuh"
#endif

///////////////////////////////////////////////////////////////////////////////
// Batched sort using device-wide CUB primitives
///////////////////////////////////////////////////////////////////////////////

template <class ind_t> __global__
void create_offsets(
    int * offsets,
    ind_t num_batches,
    ind_t batch_size) {

    const ind_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    for (ind_t id = thid; id <= num_batches; id += blockDim.x*gridDim.x) {
        offsets[id] = id*batch_size;
    }
}

template <class val_t, class ind_t> __global__
void broadcast_vector(
    val_t * __restrict__ source,       // one line of length width
    val_t * __restrict__ target,       // a matrix of dimension height x width
    const ind_t width,                 // the width
    const ind_t height) {              // the height

    const ind_t thid  = blockDim.x*blockIdx.x+threadIdx.x;

    for (ind_t j = thid; j < width; j += blockDim.x*gridDim.x) {
        val_t cache = source[j];
        for (ind_t i = 0; i < height; i++)
            target[i*width+j] = cache;
    }
}

template<
    class key_t,                       // data type for the keys
    class val_t,                       // data type for the values
    class ind_t,                       // data type for the indices
    bool transposed=false,             // wether to transpose result or not
    bool ascending=false,              // sort order of the primitive
    ind_t chunk_size=128>              // number of concurrently sorted batches
void pathway_sort(
    key_t * __restrict__ Keys,         // keys of all batches   (inplace)
    val_t * __restrict__ Source,       // values for one batch  (constin)
    val_t * __restrict__ Target,       // values of all batches (inplace)
    ind_t batch_size,                  // number of entries in a batch
    ind_t num_batches) {               // number of batches

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(device_sort_correlated_genes)
    #endif

    #ifdef CUDA_GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Sort " << num_batches << " correlation arrays "
              << std::endl << "STATUS: each of size " << batch_size
              << " on the GPU." << std::endl;
    #endif

    key_t * keys_out;
    val_t * vals_out;
    cudaMalloc(&keys_out, sizeof(key_t)*chunk_size*batch_size);          CUERR
    cudaMalloc(&vals_out, sizeof(val_t)*chunk_size*batch_size);          CUERR

    // broadcast the bitmask to all permutations
    broadcast_vector<<<32, 1024>>>(Source,
                                   Target,
                                   batch_size,
                                   num_batches);                         CUERR

    cub::DoubleBuffer<key_t> d_keys(Keys, keys_out);
    cub::DoubleBuffer<val_t> d_vals(Target, vals_out);

    void * d_temp_storage;
    size_t temp_storage_bytes;

    // create offset array
    int * offsets;
    cudaMalloc(&offsets, sizeof(int)*(chunk_size+1));                    CUERR
    create_offsets<<<32, 1024>>>(offsets,
                                 chunk_size,
                                 batch_size);                            CUERR

    // dry run
    if (ascending) {
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            d_keys,
            d_vals,
            chunk_size*batch_size,
            chunk_size,
            offsets,
            offsets+1);                                                  CUERR
    } else {
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            nullptr,
            temp_storage_bytes,
            d_keys,
            d_vals,
            chunk_size*batch_size,
            chunk_size,
            offsets,
            offsets+1);                                                  CUERR
    }

    // allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);                     CUERR

    // sort chunk_size many arrays in parallel
    const ind_t width = chunk_size*batch_size;
    for (ind_t chunk = 0; chunk < num_batches/chunk_size; chunk++) {

        // actual run
        const ind_t offset = chunk*width;
        cub::DoubleBuffer<key_t> d_keys(Keys+offset, keys_out);
        cub::DoubleBuffer<val_t> d_vals(Target+offset, vals_out);

        if (ascending) {
            cub::DeviceSegmentedRadixSort::SortPairs(
                d_temp_storage,
                temp_storage_bytes,
                d_keys,
                d_vals,
                width,
                chunk_size,
                offsets,
                offsets+1);                                              CUERR
        } else {
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp_storage,
                temp_storage_bytes,
                d_keys,
                d_vals,
                width,
                chunk_size,
                offsets,
                offsets+1);                                              CUERR
        }

        if (d_keys.Current() != Keys+offset) {
            cudaMemcpy(Keys+offset, keys_out, sizeof(key_t)*width,
                       cudaMemcpyDeviceToDevice);                        CUERR
            cudaMemcpy(Target+offset, vals_out, sizeof(val_t)*width,
                       cudaMemcpyDeviceToDevice);                        CUERR
        }
    }

    // perform the sort for the remaining vectors
    const ind_t remaining = num_batches%chunk_size;
    if (remaining) {

        const ind_t rem_width = remaining*batch_size;
        const ind_t batch = num_batches-remaining;
        const ind_t offset = batch*batch_size;

        cub::DoubleBuffer<key_t> d_keys(Keys+offset, keys_out);
        cub::DoubleBuffer<val_t> d_vals(Target+offset, vals_out);

        if (ascending) {
            cub::DeviceSegmentedRadixSort::SortPairs(
                nullptr,
                temp_storage_bytes,
                d_keys,
                d_vals,
                rem_width,
                remaining,
                offsets,
                offsets+1);                                              CUERR

            cub::DeviceSegmentedRadixSort::SortPairs(
                d_temp_storage,
                temp_storage_bytes,
                d_keys,
                d_vals,
                rem_width,
                remaining,
                offsets,
                offsets+1);                                              CUERR
        } else {
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr,
                temp_storage_bytes,
                d_keys,
                d_vals,
                rem_width,
                remaining,
                offsets,
                offsets+1);                                              CUERR

            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp_storage,
                temp_storage_bytes,
                d_keys,
                d_vals,
                rem_width,
                remaining,
                offsets,
                offsets+1);                                              CUERR
        }

        if (d_keys.Current() != Keys+offset) {
            cudaMemcpy(Keys+offset, keys_out, sizeof(key_t)*rem_width,
                       cudaMemcpyDeviceToDevice);                        CUERR
            cudaMemcpy(Target+offset, vals_out, sizeof(val_t)*rem_width,
                       cudaMemcpyDeviceToDevice);                        CUERR
        }
    }

    // get rid of the memory
    cudaFree(keys_out);                                                  CUERR
    cudaFree(vals_out);                                                  CUERR
    cudaFree(offsets);                                                   CUERR
    cudaFree(d_temp_storage);                                            CUERR

    cudaDeviceSynchronize();

    if (transposed) {
        transpose(Target, num_batches, batch_size);
        transpose(Keys, num_batches, batch_size);
    }

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(device_sort_correlated_genes)
    #endif
}

#ifdef STANDALONE_UNIT_TEST
int main(){

    typedef int val_t;
    typedef int key_t;
    typedef int ind_t;

    ind_t batch_size = 20000;
    ind_t num_batches = 20000;

    std::cout << batch_size*num_batches*(sizeof(key_t)+sizeof(ind_t))/(1<<20)
              << std::endl;

    val_t * vals, * Vals;
    val_t * orig, * Orig;
    key_t * keys, * Keys;

    cudaSetDevice(0);
    cudaDeviceReset();

    cudaMallocHost(&keys, sizeof(key_t)*num_batches*batch_size);         CUERR
    cudaMallocHost(&vals, sizeof(val_t)*num_batches*batch_size);         CUERR
    cudaMallocHost(&orig, sizeof(val_t)*batch_size);                     CUERR

    TIMERSTART(fill)
    for (ind_t batch = 0; batch < num_batches; batch++)
        for (ind_t elem = 0; elem < batch_size; elem++) {
            keys[batch*batch_size+elem] = batch_size-elem;
            vals[batch*batch_size+elem] = 0;
        }

    for (ind_t elem = 0; elem < batch_size; elem++) {
        orig[elem] = elem;
    }
    TIMERSTOP(fill)

    cudaMalloc(&Keys, sizeof(key_t)*num_batches*batch_size);             CUERR
    cudaMalloc(&Vals, sizeof(val_t)*num_batches*batch_size);             CUERR
    cudaMalloc(&Orig, sizeof(val_t)*batch_size);                         CUERR

    TIMERSTART(H2D)
    cudaMemcpy(Keys, keys, sizeof(key_t)*num_batches*batch_size,
               cudaMemcpyHostToDevice);                                  CUERR
    cudaMemcpy(Vals, vals, sizeof(val_t)*num_batches*batch_size,
               cudaMemcpyHostToDevice);                                  CUERR
    cudaMemcpy(Orig, orig, sizeof(val_t)*batch_size,
               cudaMemcpyHostToDevice);                                  CUERR
    TIMERSTOP(H2D)

    TIMERSTART(sort)
    pathway_sort<key_t, val_t, ind_t, false, true>(
        Keys,
        Orig,
        Vals,
        num_batches,
        batch_size);                                                     CUERR
    TIMERSTOP(sort)

    TIMERSTART(D2H)
    cudaMemcpy(keys, Keys, sizeof(key_t)*num_batches*batch_size,
               cudaMemcpyDeviceToHost);                                  CUERR
    cudaMemcpy(vals, Vals, sizeof(val_t)*num_batches*batch_size,
               cudaMemcpyDeviceToHost);                                  CUERR
    cudaMemcpy(orig, Orig, sizeof(val_t)*batch_size,
               cudaMemcpyDeviceToHost);                                  CUERR
    TIMERSTOP(D2H)


    #ifdef OUTPUT
    for (ind_t batch = 0; batch < num_batches; batch++)
        for (ind_t elem = 0; elem < batch_size; elem++)
            std::cout << batch << "\t" << elem << "\t"
                      << keys[batch*batch_size+elem] << "\t"
                      << vals[batch*batch_size+elem] << std::endl;
    #endif

    #ifdef CHECK
    TIMERSTART(check)
    bool check = true;
    for (ind_t batch = 0; batch < num_batches; batch++)
        check &= std::is_sorted(keys+batch*batch_size,
                                keys+(batch+1)*batch_size);
    std::cout << (check ? "sorted" : "not sorted") << std::endl;
    TIMERSTOP(check)
    #endif
}
#endif
#endif
