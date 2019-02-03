#ifndef CUDA_GSEA_BATCHED_SORT
#define CUDA_GSEA_BATCHED_SORT

// #define UNIT_TEST
#ifdef UNIT_TEST
#define CUDA_GSEA_OPENMP_ENABLED
#define CHECK
// #define OUTPUT
#endif

#include "cub/cub.cuh"
#include <iostream>
#include <algorithm>

#ifndef UNIT_TEST
#include "cuda_helpers.cuh"
#include "configuration.cuh"
#endif

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include <numeric>

#ifdef UNIT_TEST
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
    bool ascending=false,              // sort order of the primitive
    ind_t chunk_size=128>              // number of concurrently sorted batches
void pathway_sort_gpu(
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

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(device_sort_correlated_genes)
    #endif
}

template<
    class key_t,                       // data type for the keys
    class val_t,                       // data type for the values
    class ind_t,                       // data type for the indices
    bool ascending=false,              // sort order of the primitive
    ind_t chunk_size=1280>              // number of concurrently sorted batches
void pathway_sort_cpu(
    key_t * __restrict__ Keys,         // keys of all batches   (inplace)
    val_t * __restrict__ Source,       // values for one batch  (constin)
    val_t * __restrict__ Target,       // values of all batches (inplace)
    ind_t batch_size,                  // number of entries in a batch
    ind_t num_batches) {               // number of batches

    // get number of chunks
    const ind_t num_chunks = (num_batches+chunk_size-1)/chunk_size;

    #ifdef CUDA_GSEA_OPENMP_ENABLED
    # pragma omp parallel
    #endif
    for (ind_t chunk = 0; chunk < num_chunks; chunk++) {

        const ind_t lower = chunk*chunk_size;
        const ind_t upper = std::min(lower+chunk_size, num_batches);

        #ifdef CUDA_GSEA_OPENMP_ENABLED
        # pragma omp for
        #endif
        for (ind_t batch = lower; batch < upper; batch++) {

            // base index
            const ind_t base = batch*batch_size;

            // auxilliary memory
            std::vector<key_t> locKey(Keys+base, Keys+base+batch_size);
            std::vector<ind_t> locInd(batch_size);
            std::iota(locInd.data(), locInd.data()+batch_size, 0);

            // sort predicate
            auto predicate = [&] (const ind_t& lhs, const ind_t& rhs) {
                if (ascending)
                    return Keys[base+lhs] < Keys[base+rhs];
                else
                    return Keys[base+lhs] > Keys[base+rhs];
            };

            // sort indices
            std::sort(locInd.data(), locInd.data()+batch_size, predicate);

            // substitution
            for (ind_t id = 0; id < batch_size; id++) {
                Keys  [base+id] = locKey[locInd[id]];
                Target[base+id] = Source[locInd[id]];
            }
        }
    }
}

#ifdef UNIT_TEST

uint32_t nvidia_hash(uint32_t x) {

    x = (x + 0x7ed55d16) + (x << 12);
    x = (x ^ 0xc761c23c) ^ (x >> 19);
    x = (x + 0x165667b1) + (x <<  5);
    x = (x + 0xd3a2646c) ^ (x <<  9);
    x = (x + 0xfd7046c5) + (x <<  3);
    x = (x ^ 0xb55a4f09) ^ (x >> 16);

    return x;
}

void check_gpu(){

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
            keys[batch*batch_size+elem] =  nvidia_hash(elem);
            vals[batch*batch_size+elem] = ~nvidia_hash(elem);
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
    pathway_sort_gpu<key_t, val_t, ind_t, true>(
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

void check_cpu(){

    typedef int val_t;
    typedef int key_t;
    typedef int ind_t;

    ind_t batch_size = 20000;
    ind_t num_batches = 20000;

    std::cout << batch_size*num_batches*(sizeof(key_t)+sizeof(ind_t))/(1<<20)
              << std::endl;

    val_t * vals;
    val_t * orig;
    key_t * keys;

    cudaMallocHost(&keys, sizeof(key_t)*num_batches*batch_size);         CUERR
    cudaMallocHost(&vals, sizeof(val_t)*num_batches*batch_size);         CUERR
    cudaMallocHost(&orig, sizeof(val_t)*batch_size);                     CUERR

    TIMERSTART(fill)
    for (ind_t batch = 0; batch < num_batches; batch++)
        for (ind_t elem = 0; elem < batch_size; elem++) {
            keys[batch*batch_size+elem] =  nvidia_hash(elem);
            vals[batch*batch_size+elem] = ~nvidia_hash(elem);
        }

    for (ind_t elem = 0; elem < batch_size; elem++) {
        orig[elem] = elem;
    }
    TIMERSTOP(fill)


    TIMERSTART(sort)
    pathway_sort_cpu<key_t, val_t, ind_t, true>(
        keys,
        orig,
        vals,
        num_batches,
        batch_size);                                                     CUERR
    TIMERSTOP(sort)


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

int main() {
    check_gpu();
    check_cpu();
}

#endif
#endif
