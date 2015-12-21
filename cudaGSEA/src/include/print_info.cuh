#ifndef CUDA_GSEA_PRINT_INFO_CUH
#define CUDA_GSEA_PRINT_INFO_CUH

#include <iostream>
#include <string>

void print_welcome() {

    std::cout << "INFO:                    __      ___________ _________ "
              << std::endl;
    std::cout << "INFO:    _______  ______/ /___ _/ ____/ ___// ____/   |"
              << std::endl;
    std::cout << "INFO:   / ___/ / / / __  / __ `/ / __ \\__ \\/ __/ / /| |"
              << std::endl;
    std::cout << "INFO:  / /__/ /_/ / /_/ / /_/ / /_/ /___/ / /___/ ___ |"
              << std::endl;
    std::cout << "INFO:  \\___/\\__,_/\\__,_/\\__,_/\\____//____/_____/_/  |_|"
              << std::endl;
    std::cout << "INFO: +------------------------------------------------+"
              << std::endl;
    std::cout << "INFO: | Parallel and Distributed Architectures Group   |"
              << std::endl;
    std::cout << "INFO: | Scientific Computing and Bioinformatics Group  |"
              << std::endl;
    std::cout << "INFO: | Johannes Gutenberg University, Mainz (Germany) |"
              << std::endl;
    std::cout << "INFO: +------------------------------------------------+"
                        << std::endl;

    std::cout << "INFO:" << std::endl;
}

template <class index_t>
void print_gsea_info(index_t num_patients,
                     index_t num_type_A,
                     index_t num_type_B,
                     index_t num_genes,
                     index_t num_paths,
                     index_t num_perms,
                     std::string metric,
                     std::string gct_filename,
                     std::string cls_filename,
                     std::string gmt_filename,
                     bool sort_direction,
                     bool swap_labels) {

    // a fancy bar suddenly appears in the wild:
    std::string bar = "===================";

    std::cout << "INFO: " << bar << bar << bar << std::endl;
    std::cout << "INFO: | The setting is the following:" << std::endl;
    std::cout << "INFO: | " << std::endl;
    std::cout << "INFO: | expression data: " << gct_filename << std::endl;
    std::cout << "INFO: | phenotype labels: " << cls_filename << std::endl;
    std::cout << "INFO: | pathways data: " << gmt_filename << std::endl;
    std::cout << "INFO: | number of patients: " << num_patients << std::endl;
    std::cout << "INFO: | number of phenotype 0: " << num_type_A << std::endl;
    std::cout << "INFO: | number of phenotype 1: " << num_type_B << std::endl;
    std::cout << "INFO: | number of unique genes: " << num_genes << std::endl;
    std::cout << "INFO: | number of pathways: " << num_paths << std::endl;
    std::cout << "INFO: | number of permutations: " << num_perms << std::endl;
    std::cout << "INFO: | used ranking metric: " << metric << std::endl;
    std::cout << "INFO: | sort direction used during ranking: " <<
                 (sort_direction ? "descending" : "ascending") << std::endl;
    std::cout << "INFO: | swapped labels (1 <-> 0): " <<
                 (swap_labels ? "true" : "false (default)") << std::endl;
    std::cout << "INFO: " << bar << bar << bar << std::endl;

}

template <class index_t>
void print_scheduler_info(index_t batch_size_perms,
                          index_t num_batches_perms,
                          index_t batch_size_paths,
                          index_t num_batches_paths) {

    // a fancy bar suddenly appears in the wild:
    std::string bar = "===================";

    std::cout << "INFO: " << bar << bar << bar << std::endl;
    std::cout << "INFO: | The scheduler determined the following plan:"
              << std::endl;
    std::cout << "INFO: | " << std::endl;
    std::cout << "INFO: | permutations batch size: " << batch_size_perms
              << std::endl;
    std::cout << "INFO: | number of permutation batches: " << num_batches_perms
              << std::endl;
    std::cout << "INFO: | pathway batch size: " << batch_size_paths
              << std::endl;
    std::cout << "INFO: | number of pathway batches: " << num_batches_paths
              << std::endl;
    std::cout << "INFO: | allover batches to be computed: "
              << num_batches_perms*num_batches_paths << std::endl;
    std::cout << "INFO: " << bar << bar << bar << std::endl;

}

template <class index_t>
void print_batch_info(index_t lower_pa,
                      index_t upper_pa,
                      index_t width_pa,
                      index_t lower_pi,
                      index_t upper_pi,
                      index_t width_pi) {

    // a fancy bar suddenly appears in the wild:
    std::string bar = "===================";

    std::cout << "INFO: " << bar << bar << bar << std::endl;
    std::cout << "INFO: | Process genes for " << width_pa
              << " paths from " << lower_pa << " up to " << upper_pa
              << std::endl << "INFO: | and " << width_pi
              << " permutations from " << lower_pi << " up to "
              << upper_pi << std::endl;
    std::cout << "INFO: " << bar << bar << bar << std::endl;

}

void print_cuda_finished() {

    // a fancy bar suddenly appears in the wild:
    std::string bar = "===================";

    std::cout << "INFO: " << bar << bar << bar << std::endl;
    std::cout << "INFO: | finished computation on the GPU" << std::endl;
    std::cout << "INFO: " << bar << bar << bar << std::endl;

}

#endif
