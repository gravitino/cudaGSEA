#ifndef CUDA_GSEA_TRANSPOSE_CUH
#define CUDA_GSEA_TRANSPOSE_CUH

#include "cuda_helpers.cuh"
#include "configuration.cuh"

#define TRANSPOSE_TILE (16)

template <class value_t, class index_t> __global__
void transpose_kernel(value_t * In,
                      value_t * Out,
                      index_t height,
                      index_t width) {

    __shared__ value_t tile[TRANSPOSE_TILE][TRANSPOSE_TILE+1];

    index_t x = blockIdx.x*TRANSPOSE_TILE + threadIdx.x;
    index_t y = blockIdx.y*TRANSPOSE_TILE + threadIdx.y;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = In[y*width+x];
    __syncthreads();

    x = blockIdx.y*TRANSPOSE_TILE + threadIdx.x;
    y = blockIdx.x*TRANSPOSE_TILE + threadIdx.y;

    if (y < width && x < height)
        Out[y*height+x] = tile[threadIdx.x][threadIdx.y];
}

template <class value_t, class index_t> __forceinline__ __host__
void transpose_store(value_t * In,
                     value_t * Out,
                     index_t height,
                     index_t width) {

    dim3 blockdim((width+TRANSPOSE_TILE-1)/TRANSPOSE_TILE,
                  (height+TRANSPOSE_TILE-1)/TRANSPOSE_TILE, 1);
    dim3 threaddim(TRANSPOSE_TILE, TRANSPOSE_TILE, 1);
    transpose_kernel<<<blockdim, threaddim>>>(In, Out, height, width);    CUERR
}

template <class value_t, class index_t> __forceinline__ __host__
void transpose(value_t * In_Out,
               index_t height,
               index_t width) {

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(device_transpose_matrix)
    #endif

    #ifdef CUDA_GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Transposing a " << height << " x " << width
              << " matrix of size " << (height*width*sizeof(value_t))
              << " bytes on the GPU." << std::endl;
    #endif

    value_t * Temp = nullptr;
    cudaMalloc(&Temp, sizeof(value_t)*height*width);                      CUERR

    transpose_store(In_Out, Temp, height, width);                         CUERR

    cudaMemcpy(In_Out, Temp, sizeof(value_t)*height*width,
               cudaMemcpyDeviceToDevice);                                 CUERR
    cudaFree(Temp);                                                       CUERR

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(device_transpose_matrix)
    #endif
}


#endif
