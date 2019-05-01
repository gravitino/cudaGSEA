#ifndef CUDA_GSEA_STATISTICS_CUH
#define CUDA_GSEA_STATISTICS_CUH

#include <vector>

template <class enrch_t, class index_t>
index_t pos_tail_count(const enrch_t * data,
                       const index_t length,
                       const enrch_t value) {

    index_t counter = 0;
    for (auto entry_ptr = data; entry_ptr != data+length; entry_ptr++)
        if (*entry_ptr >= value)
            counter++;

    return counter;
}


template <class enrch_t, class index_t>
index_t neg_tail_count(const enrch_t * data,
                       const index_t length,
                       const enrch_t value) {

    index_t counter = 0;
    for (auto entry_ptr = data; entry_ptr != data+length; entry_ptr++)
        if (*entry_ptr <= value)
            counter++;

    return counter;
}

template <class enrch_t, class index_t>
index_t two_tail_count(const enrch_t * data,
                       const index_t length,
                       const enrch_t value) {

    index_t counter = 0;
    for (auto entry_ptr = data; entry_ptr != data+length; entry_ptr++)
        if (std::abs(*entry_ptr) >= std::abs(value))
            counter++;

    return counter;
}

template <class enrch_t, class index_t>
enrch_t compute_fdr_q(const enrch_t * scores,
                      const enrch_t * random_scores,
                      const index_t num_paths,
                      const index_t num_perms,
                      const index_t index_to_ignore,  // index of random scores corresponding to score
                      const enrch_t score) {

    // more extreme than observed NES* in sets (S)
    enrch_t more_extreme_observed = 0;
    // number of S such that NES(S) ≥ 0 when NES* ≥ 0 [or such that NES(S) ≤ 0 when NES* ≤ 0]
    enrch_t considered_sets = 0;

    // more extreme than observed NES* in all permutations π (that are being considered for the positive/negative score)
    enrch_t more_extreme_random_total = 0;
    // temporary: more extreme than observed NES* in permutations π of gene set S
    enrch_t more_extreme_random = 0;
    // temporary: count of all π for given S with NES(S, π) ≥ 0 [or NES(S) ≤ 0]
    enrch_t random_count = 0;
    // count of all S with π such that there is at least one NES(S, π) ≥ 0 [or NES(S) ≤ 0]
    enrch_t non_empty_sets_count = 0;

    for (index_t other = 0; other < num_paths; other++) {

        if (score >= 0) {

            // count of NES(S, π) such that: NES(S, π) ≥ NES*
            more_extreme_random = pos_tail_count(&random_scores[other*num_perms], num_perms, score);

            // count of π for given S, such that NES(S, π) ≥ 0
            random_count = pos_tail_count(&random_scores[other*num_perms], num_perms, (enrch_t) 0.0);

            // count NES(S) such that NES(S) ≥ NES*
            if (scores[other] >= score)
                more_extreme_observed++;

            // count of observed S with NES(S) ≥ 0
            if (scores[other] >= 0)
                considered_sets++;
        }
        else if (score < 0) {
            more_extreme_random = neg_tail_count(&random_scores[other*num_perms], num_perms, score);

            random_count = neg_tail_count(&random_scores[other*num_perms], num_perms, (enrch_t) 0.0);

            if (scores[other] <= score)
                more_extreme_observed++;

            if (scores[other] <= 0)
                considered_sets++;
        }

        if (random_count != 0) {
            non_empty_sets_count++;
            more_extreme_random_total += more_extreme_random / random_count;
        }

    }

    // the percentage of all (S, π) with NES(S, π) ≥ 0, whose NES(S, π) ≥ NES*,
    enrch_t numerator = more_extreme_random_total / non_empty_sets_count;
    // the percentage of observed S with NES(S) ≥ 0, whose NES(S) ≥ NES*
    enrch_t denominator = more_extreme_observed / considered_sets;

    if (denominator == 0)
        return 0;
    else
    {
        enrch_t q = numerator / denominator;

        // q-value is an estimate of FDR so it does not have to be strictly lower than 1;
        // both GSEArot and gsea-desktop trim q-value at one, so let's do the same:

        if(q > 1)
            return 1;

        return q;
    }

}


template <class enrch_t, class index_t> __host__
std::vector<enrch_t> final_statistics(std::vector<enrch_t>& enrchScores,
                                      const index_t num_paths,
                                      const index_t num_perms) {

    // 5 since (ES, NES, p-value, FWER, FDR) for each path
    // data is stored in a struct of arrays
    std::vector<enrch_t> result(5*num_paths, 0);

    // first of all store all enrichment scores
    for (index_t path = 0; path < num_paths; path++)
        result[path] = enrchScores[path*num_perms];

    // now compute the normalized enrichment score for each path
    for (index_t path = 0; path < num_paths; path++) {
        enrch_t sum_positive = 0;
        enrch_t sum_negative = 0;
        enrch_t cnt_positive = 0;
        enrch_t cnt_negative = 0;

        // TODO: maybe make this Kahan stable
        for (index_t perm = 0; perm < num_perms; perm++) {
            enrch_t value = enrchScores[path*num_perms+perm];
            if (value > 0) {
                sum_positive += value;
                cnt_positive++;
            }
            else {
                sum_negative -= value;
                cnt_negative++;
            }
        }

        enrch_t factor_positive = cnt_positive / sum_positive;
        enrch_t factor_negative = cnt_negative / sum_negative;

        if (result[path] >= 0)
            result[1*num_paths+path] = result[path] * factor_positive;
        else
            result[1*num_paths+path] = result[path] * factor_negative;

        // dump to result:
        // a memory-saving trick (this is not the final purpose of these columns)
        result[3*num_paths+path] = factor_positive;
        result[4*num_paths+path] = factor_negative;


        // Interestingly, the joined-normalisation produced results
        // closer to those from GSEA desktop in some comparisons...
        // leaving the code here for now - just in case.
        /*
        enrch_t sum = 0;
        for (index_t perm = 0; perm < num_perms; perm++)
            sum += std::abs(enrchScores[path*num_perms+perm]);
        enrch_t factor = num_perms / sum;
        result[1*num_paths+path] = result[path]*factor;
        result[3*num_paths+path] = factor;
        result[4*num_paths+path] = factor;
        */
    }

    // compute nominal p-values for each path
    for (index_t path = 0; path < num_paths; path++) {
        enrch_t * data = enrchScores.data()+path*num_perms;
        enrch_t p_value = two_tail_count(data, num_perms, result[path]);
        result[2*num_paths+path] = p_value/num_perms;
    }

    // permutation-wise extrema
    auto infty = std::numeric_limits<enrch_t>::infinity();
    std::vector<enrch_t> max_vals(num_perms, -infty);
    std::vector<enrch_t> min_vals(num_perms, +infty);

    // renormalize all enrichment scores and determine extrema
    for (index_t path = 0; path < num_paths; path++) {
        enrch_t factor_positive = result[3*num_paths+path];
        enrch_t factor_negative = result[4*num_paths+path];

        for (index_t perm = 0; perm < num_perms; perm++) {

            enrch_t value = enrchScores[path*num_perms+perm];

            // normalize enrichment scores
            if (value >= 0)
                value *= factor_positive;
            else
                value *= factor_negative;

            enrchScores[path*num_perms+perm] = value;

            // update extrema
            max_vals[perm] = max_vals[perm] < value ? value : max_vals[perm];
            min_vals[perm] = min_vals[perm] > value ? value : min_vals[perm];
        }
    }

    // compute family-wise error
    for (index_t path = 0; path < num_paths; path++) {
        enrch_t score = result[num_paths+path];
        enrch_t p_value = 0;

        if (score > 0)
            p_value = pos_tail_count(max_vals.data(), num_perms, score);
        else
            p_value = neg_tail_count(min_vals.data(), num_perms, score);

        // not by num_perms but by tail size
        result[3*num_paths+path] = p_value/num_perms;
    }

    // FDR-Q: computed for all scores.
    for (index_t path = 0; path < num_paths; path++) {

        // normalized enrichment score
        enrch_t score = result[num_paths+path];

        result[4*num_paths+path] = compute_fdr_q(
            &result[num_paths],       // real scores
            enrchScores.data(),       // random scores
            num_paths,
            num_perms,
            path,
            score
        );
    }

    return result;
}


#endif
