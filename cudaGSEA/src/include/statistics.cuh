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
        enrch_t sum = 0;
        // TODO: maybe make this Kahan stable
        for (index_t perm = 0; perm < num_perms; perm++)
            sum += std::abs(enrchScores[path*num_perms+perm]);

        // dump to result
        enrch_t factor = num_perms / sum;
        result[1*num_paths+path] = result[path]*factor;
        result[3*num_paths+path] = factor;
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
        enrch_t factor = result[3*num_paths+path];
        for (index_t perm = 0; perm < num_perms; perm++) {

            // normalize enrichment scores
            enrch_t value = enrchScores[path*num_perms+perm]*factor;
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

        result[3*num_paths+path] = p_value/num_perms;
    }

    // FDR-Q
    for (index_t path = 0; path < num_paths; path++) {
        // normalized enrichment score
        enrch_t score = result[num_paths+path];
        enrch_t more_extreme_observed = 0;
        enrch_t more_extreme_random = 0;

        for (index_t other = 0; other < num_paths; other++) {
            if (path == other)
                continue;

            if (score == 0) {
                continue;
            }
            else if (score > 0) {
                more_extreme_random += pos_tail_count(&enrchScores[other*num_perms], num_perms, score);
                if (result[num_paths+other] >= score)
                    more_extreme_observed += 1;
            }
            else {
                more_extreme_random += neg_tail_count(&enrchScores[other*num_perms], num_perms, score);
                if (result[num_paths+other] <= score)
                    more_extreme_observed += 1;
            }

        }
        enrch_t nominator = more_extreme_random / ((num_paths - 1) * num_perms);
        enrch_t denominator = more_extreme_observed / (num_paths - 1);

        if (denominator == 0)
            result[4*num_paths+path] = 0;
        else
            result[4*num_paths+path] = nominator / denominator;
    }

    return result;
}


#endif
