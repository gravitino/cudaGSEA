#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <iostream>

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

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
        std::cout << "TIMING: " << time##label << " ms (" << #label << ")"   \
                  << std::endl;

// transfer constants
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define H2H (cudaMemcpyHostToHost)
#define D2D (cudaMemcpyDeviceToDevice)

// transfer to GPU
template <class pointer_t, class index_t>
pointer_t * toGPU(pointer_t * input, index_t size) {
    pointer_t * Input = nullptr;
    cudaMalloc(&Input, sizeof(pointer_t)*size);                           CUERR
    cudaMemcpy(Input, input, sizeof(pointer_t)*size, H2D);                CUERR
    return Input;
}

// plain GPU array
template <class pointer_t, class index_t>
pointer_t * zeroGPU(index_t size) {
    pointer_t * Empty = nullptr;
    cudaMalloc(&Empty, sizeof(pointer_t)*size);                           CUERR
    cudaMemset(Empty, 0, sizeof(pointer_t)*size);                         CUERR
    return Empty;
}

// copy GPU array
template <class pointer_t, class index_t>
pointer_t * cpyGPU(pointer_t * input, index_t size) {
    pointer_t * Copy = nullptr;
    cudaMalloc(&Copy, sizeof(pointer_t)*size);                            CUERR
    cudaMemcpy(Copy, input, sizeof(pointer_t)*size, D2D);                 CUERR
    return Copy;
}

// get free RAM of GPU
size_t freeRAM() {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);                                        CUERR
    return free;
}

// range kernel
template <class value_t, class index_t> __global__
void range_kernel(value_t * array, index_t length) {
    
    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    if (thid < length)
        array[thid] = thid;
}

template <class value_t, class index_t>
void range(value_t * array, index_t length) {
    range_kernel<<<SDIV(length, 1024), 1024>>>(array, length);            CUERR
}






#endif
