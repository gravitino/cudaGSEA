#ifndef CUDA_GSEA_FUNCTORS
#define CUDA_GSEA_FUNCTORS

#include "math_helpers.cuh"

//////////////////////////////////////////////////////////////////////////////
// low-level combiner
//////////////////////////////////////////////////////////////////////////////

struct combine_difference {
    template <class value_t> __host__ __device__ __forceinline__
    value_t operator()(const value_t& lhs, const value_t& rhs) const {
        return lhs-rhs;
    }
};

struct combine_quotient {
    template <class value_t> __host__ __device__ __forceinline__
    value_t operator()(const value_t& lhs, const value_t& rhs) const {
        return lhs/rhs;
    }
};

struct combine_log2_quotient {
    template <class value_t> __host__ __device__ __forceinline__
    value_t operator()(const value_t& lhs, const value_t& rhs) const {
        return culog(lhs/rhs)*CUDA_GSEA_INV_LOG2;
    }
};

template <bool fixlow=true>
struct combine_signal2noise {
    template <class value_t, class index_t> __host__ __device__ __forceinline__
    value_t operator()(const value_t& mean_A,
                       const value_t& var_A,
                       const index_t& num_type_A,
                       const value_t& mean_B,
                       const value_t& var_B,
                       const index_t& num_type_B) const {

        if (!fixlow) {
            return (mean_A-mean_B)/(cusqrt(var_A)+cusqrt(var_B));
        } else {
            value_t sigma_A = cusqrt(var_A);
            value_t sigma_B = cusqrt(var_B);
            value_t minallowed_A = 0.2*cuabs(mean_A);
            value_t minallowed_B = 0.2*cuabs(mean_B);
            sigma_A = minallowed_A < sigma_A ? sigma_A : minallowed_A;
            sigma_B = minallowed_B < sigma_B ? sigma_B : minallowed_B;

            return (mean_A-mean_B)/(sigma_A+sigma_B);
        }
    }
};


template <bool fixlow=true>
struct combine_T_test {
    template <class value_t, class index_t> __host__ __device__ __forceinline__
    value_t operator()(const value_t& mean_A,
                       const value_t& var_A,
                       const index_t& num_type_A,
                       const value_t& mean_B,
                       const value_t& var_B,
                       const index_t& num_type_B) const {

        if (!fixlow) {
            return (mean_A-mean_B)/cusqrt(var_A/num_type_A+var_B/num_type_B);
        } else {
            value_t va_A = var_A;
            value_t va_B = var_B;
            value_t minallowed_A = 0.04*cuabs(mean_A);
            value_t minallowed_B = 0.04*cuabs(mean_B);
            va_A = minallowed_A < va_A ? va_A : minallowed_A;
            va_B = minallowed_B < va_B ? va_B : minallowed_B;

            return (mean_A-mean_B)/cusqrt(va_A/num_type_A+va_B/num_type_B);
        }
    }
};


//////////////////////////////////////////////////////////////////////////////
// low-level accumulators
//////////////////////////////////////////////////////////////////////////////

template <class funct_t, bool transposed=true>
struct accumulate_naive_mean {

    const funct_t combine;

    __host__ __device__
    accumulate_naive_mean(funct_t combine_) : combine(combine_) {}

    template <class label_t, class index_t, class value_t>
    __host__ __device__ __forceinline__
    value_t operator()(label_t * labels,
                       value_t * table,
                       index_t lane,
                       index_t gene,
                       index_t num_genes,
                       index_t num_type_A,
                       index_t num_type_B) {

        // compute mean of phenotypes using naive sum
        value_t mean_A = 0, mean_B = 0;
        for (index_t id = 0; id < lane; id++) {

            // cache label and value
            const label_t label = !labels[id];
            const value_t value = transposed ? table[id*num_genes+gene] :
                                               table[gene*lane+id];

            // add up the contributions
            mean_A +=  label*value;
            mean_B += !label*value;
        }

        mean_A /= num_type_A;
        mean_B /= num_type_B;

        return combine(mean_A, mean_B);
    }
};

template <class funct_t, bool transposed=true>
struct accumulate_kahan_mean {

    const funct_t combine;

    __host__ __device__
    accumulate_kahan_mean(funct_t combine_) : combine(combine_) {}

    template <class label_t, class index_t, class value_t>
    __host__ __device__ __forceinline__
    value_t operator()(label_t * labels,
                       value_t * table,
                       index_t lane,
                       index_t gene,
                       index_t num_genes,
                       index_t num_type_A,
                       index_t num_type_B) {

        // compute mean of phenotypes using Kahan stable sum
        value_t mean_A = 0, mean_B = 0, kahan_A = 0, kahan_B = 0;
        for (index_t id = 0; id < lane; id++) {

            // cache label and value
            const label_t label = !labels[id];
            const value_t value = transposed ? table[id*num_genes+gene] :
                                               table[gene*lane+id];

            // Kahan summation
            const value_t accum = label ? mean_A : mean_B;
            const value_t alpha = value - (label ? kahan_A : kahan_B);
            const value_t beta  = accum + alpha;
            const value_t kahan = (beta - accum) - alpha;

            // update states
            kahan_A =  label ? kahan : kahan_A;
            kahan_B = !label ? kahan : kahan_B;
            mean_A  =  label ? beta  : mean_A;
            mean_B  = !label ? beta  : mean_B;

            // this is what really happens
            // mean_A +=  label*value;
            // mean_B += !label*value;
        }


        mean_A /= num_type_A;
        mean_B /= num_type_B;

        return combine(mean_A, mean_B);
    }
};

template <class funct_t, bool transposed=true, bool biased=true>
struct accumulate_onepass_stats {

    const funct_t combine;

    __host__ __device__
    accumulate_onepass_stats(funct_t combine_) : combine(combine_) {}

    template <class label_t, class index_t, class value_t>
    __host__ __device__ __forceinline__
    value_t operator()(label_t * labels,
                       value_t * table,
                       index_t lane,
                       index_t gene,
                       index_t num_genes,
                       index_t num_type_A,
                       index_t num_type_B) {

        // compute mean and variance of phenotypes using naive sum
        value_t mean_A = 0, mean_B = 0 , var_A = 0, var_B = 0;
        for (index_t id = 0; id < lane; id++) {

            // cache label and value
            const label_t label = !labels[id];
            const value_t value = transposed ? table[id*num_genes+gene] :
                                               table[gene*lane+id];

            // add up the contributions
            mean_A =  label ? mean_A + value : mean_A;
            mean_B = !label ? mean_B + value : mean_B;
            var_A =  label ? var_A + value*value : var_A;
            var_B = !label ? var_B + value*value : var_B;
        }

        mean_A /= num_type_A;
        mean_B /= num_type_B;

        // this may cause massive cancelation
        if (biased) {
            var_A = var_A/num_type_A - mean_A*mean_A;
            var_B = var_B/num_type_B - mean_B*mean_B;
        } else {
            var_A = var_A/(num_type_A-1) -
                    mean_A*mean_A*num_type_A/(num_type_A-1);
            var_B = var_B/(num_type_B-1) -
                    mean_B*mean_B*num_type_B/(num_type_B-1);
        }

        return combine(mean_A, var_A, num_type_A,
                       mean_B, var_B, num_type_B);
    }
};


template <class funct_t, bool transposed=true, bool biased=true>
struct accumulate_twopass_stats {

    const funct_t combine;

    __host__ __device__
    accumulate_twopass_stats(funct_t combine_) : combine(combine_) {}

    template <class label_t, class index_t, class value_t>
    __host__ __device__ __forceinline__
    value_t operator()(label_t * labels,
                       value_t * table,
                       index_t lane,
                       index_t gene,
                       index_t num_genes,
                       index_t num_type_A,
                       index_t num_type_B) {

        // compute mean of phenotypes using naive sum
        value_t mean_A = 0, mean_B = 0;
        for (index_t id = 0; id < lane; id++) {

            // cache label and value
            const label_t label = !labels[id];
            const value_t value = transposed ? table[id*num_genes+gene] :
                                               table[gene*lane+id];

            mean_A  =  label ? mean_A + value  : mean_A;
            mean_B  = !label ? mean_B + value  : mean_B;

            // this is what really happens
            // mean_A +=  label*value;
            // mean_B += !label*value;
        }

        mean_A /= num_type_A;
        mean_B /= num_type_B;

        // compute mean of phenotypes using naive sum
        value_t var_A = 0, var_B = 0;
        for (index_t id = 0; id < lane; id++) {

            // cache label and value
            const label_t label = !labels[id];
            const value_t value = transposed ? table[id*num_genes+gene] :
                                               table[gene*lane+id];

            // add up the contributions
            var_A =  label ? var_A + (value-mean_A)*(value-mean_A) : var_A;
            var_B = !label ? var_B + (value-mean_B)*(value-mean_B) : var_B;
        }

        if (biased) {
            var_A = var_A/num_type_A;
            var_B = var_B/num_type_B;
        } else {
            var_A = var_A/(num_type_A-1);
            var_B = var_B/(num_type_B-1);
        }

        return combine(mean_A, var_A, num_type_A,
                       mean_B, var_B, num_type_B);
    }
};

// http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
template <class funct_t, bool transposed=true, bool biased=true>
struct accumulate_overkill_stats {

    const funct_t combine;

    __host__ __device__
    accumulate_overkill_stats(funct_t combine_) : combine(combine_) {}

    template <class label_t, class index_t, class value_t>
    __host__ __device__ __forceinline__
    value_t operator()(label_t * labels,
                       value_t * table,
                       index_t lane,
                       index_t gene,
                       index_t num_genes,
                       index_t num_type_A,
                       index_t num_type_B) {

        // compute mean of phenotypes using Kahan stable sum
        value_t mean_A = 0, mean_B = 0, kahan_A = 0, kahan_B = 0;
        for (index_t id = 0; id < lane; id++) {

            // cache label and value
            const label_t label = !labels[id];
            const value_t value = transposed ? table[id*num_genes+gene] :
                                               table[gene*lane+id];

            // Kahan summation
            const value_t accum = label ? mean_A : mean_B;
            const value_t alpha = value - (label ? kahan_A : kahan_B);
            const value_t beta  = accum + alpha;
            const value_t kahan = (beta - accum) - alpha;

            // update states
            kahan_A =  label ? kahan : kahan_A;
            kahan_B = !label ? kahan : kahan_B;
            mean_A  =  label ? beta  : mean_A;
            mean_B  = !label ? beta  : mean_B;

            // this is what really happens
            // mean_A +=  label*value;
            // mean_B += !label*value;
        }

        mean_A /= num_type_A;
        mean_B /= num_type_B;

        // compute mean of phenotypes using compensated sum
        value_t var_A = 0, var_B = 0, aux_A = 0, aux_B = 0;
        for (index_t id = 0; id < lane; id++) {

            // cache label and value
            const label_t label = !labels[id];
            const value_t value = transposed ? table[id*num_genes+gene] :
                                               table[gene*lane+id];

            // add up the contributions
            aux_A =  label ? aux_A + (value-mean_A) : aux_A;
            aux_B = !label ? aux_B + (value-mean_B) : aux_B;
            var_A =  label ? var_A + (value-mean_A)*(value-mean_A) : var_A;
            var_B = !label ? var_B + (value-mean_B)*(value-mean_B) : var_B;
        }

        var_A = var_A - aux_A*aux_A/num_type_A;
        var_B = var_B - aux_B*aux_B/num_type_B;

        if (biased) {
            var_A = var_A/num_type_A;
            var_B = var_B/num_type_B;
        } else {
            var_A = var_A/(num_type_A-1);
            var_B = var_B/(num_type_B-1);
        }

        return combine(mean_A, var_A, num_type_A,
                       mean_B, var_B, num_type_B);
    }
};

template <class funct_t, bool transposed=true, bool biased=true>
struct accumulate_knuth_stats {

    const funct_t combine;

    __host__ __device__
    accumulate_knuth_stats(funct_t combine_) : combine(combine_) {}

    template <class label_t, class index_t, class value_t>
    __host__ __device__ __forceinline__
    value_t operator()(label_t * labels,
                       value_t * table,
                       index_t lane,
                       index_t gene,
                       index_t num_genes,
                       index_t num_type_A,
                       index_t num_type_B) {

        // compute mean and variance of phenotypes using naive sum
        value_t mean_A = 0, mean_B = 0 , var_A = 0, var_B = 0;
        index_t iter_A = 0, iter_B = 0;
        for (index_t id = 0; id < lane; id++) {

            // cache label and value
            const label_t label = !labels[id];
            const value_t value = transposed ? table[id*num_genes+gene] :
                                               table[gene*lane+id];

            // count labels
            iter_A =  label ? iter_A+1 : iter_A;
            iter_B = !label ? iter_B+1 : iter_B;

            // auxilliary residue
            const value_t delta = label ? value-mean_A : value-mean_B;

            // update mean
            mean_A =  label ? mean_A + delta/iter_A : mean_A;
            mean_B = !label ? mean_B + delta/iter_B : mean_B;

            // update variance
            var_A =  label ? var_A + delta*(value-mean_A) : var_A;
            var_B = !label ? var_B + delta*(value-mean_B) : var_B;
        }

        // this may cause massive cancelation
        if (biased) {
            var_A = var_A/num_type_A;
            var_B = var_B/num_type_B;
        } else {
            var_A = var_A/(num_type_A-1);
            var_B = var_B/(num_type_B-1);
        }

        return combine(mean_A, var_A, num_type_A,
                       mean_B, var_B, num_type_B);
    }
};


#endif
