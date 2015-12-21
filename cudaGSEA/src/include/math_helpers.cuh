#ifndef CUDA_GSEA_MATH_HELPERS_CUH
#define CUDA_GSEA_MATH_HELPERS_CUH

#ifndef CUDA_GSEA_INV_LOG2
#define CUDA_GSEA_INV_LOG2 (1.44269504089)
#endif

// default math

template <class T> __host__ __device__ __forceinline__
T cuabs(const T& x) {
    return x < 0 ? -x : x;
}

template <class T> __host__ __device__ __forceinline__
T cumin(const T& x, const T& y) {
    return x < y ? x : y;
}

template <class T> __host__ __device__ __forceinline__
T cumax(const T& x, const T& y) {
    return x < y ? y : x;
}

// specializations for uniform code

__host__ __device__ __forceinline__
float cusqrt(const float& x) {
    return sqrtf(x);
}

__host__ __device__ __forceinline__
double cusqrt(const double& x) {
    return sqrt(x);
}

__host__ __device__ __forceinline__
float culog(const float& x) {
    return logf(x);
}

__host__ __device__ __forceinline__
double culog(const double& x) {
    return log(x);
}

#endif
