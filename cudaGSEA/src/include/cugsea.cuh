#ifndef CUDA_GSEA_CORE_CUH
#define CUDA_GSEA_CORE_CUH

#include <iostream>                         // std::cout
#include <vector>                           // std::vector
#include <iomanip>                          // std::setprecision

// CUDA helpers
#include "cuda_helpers.cuh"                 // convenience cmds, errors

// CUDA GSEA related stuff
#include "enrichment_scores.cuh"            // compute enrichments
#include "correlate_genes.cuh"              // correlate genes
#include "read_data_files.cuh"              // read gmt, gct and cls
#include "batch_scheduler.cuh"              // compute batches if ram too small
#include "create_bitmaps.cuh"               // pathway representation
#include "configuration.cuh"                // status messages and timings
#include "batched_sort.cuh"                 // sort a bunch of arrays
#include "copy_strided.cuh"                 // strided copy of results
#include "bitmap_types.cuh"                 // bitmap types for pathways
#include "dump_binary.cuh"                  // write result to disk
#include "statistics.cuh"                   // final statistics
#include "print_info.cuh"                   // information of the gsea routine
#include "transpose.cuh"                    // transposition of matrices

template <class exprs_t,                    // data type for expression data
          class index_t,                    // data type for indexing
          class label_t,                    // data type for storing labels
          class bitmp_t,                    // data type for storing pathways
          class enrch_t> __host__           // data type for enrichment scores
void compute_gsea(std::string gct_filename, // name of the expression data file
                  std::string cls_filename, // name of the class label file
                  std::string gmt_filename, // name of the pathway data file
                  std::string metric,       // ranking metric specifier
                  index_t num_perms,        // number of permutations (incl. id)
                  bool sort_direction=true, // 1: descending, 0: ascending
                  bool swap_labels=false,   // 1: swap phenotypes, 0: as is
                  std::string dump_filename="") {

    // make sure the user does not use bool as label_t to avoid problems
    static_assert(!std::is_same<label_t, bool>::value,
                  "You can't use bool for label_t! Use unsigned char instead.");

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_and_device_overall)
    #endif

    #ifdef CUDA_GSEA_PRINT_WARNINGS
    if (sizeof(label_t) > 1)
        std::cout << "WARNING: Choose a narrower label_t, e.g. char."
                  << std::endl;
    if (!sort_direction)
        std::cout << "WARNING: You chose a non default sort direction."
                  << std::endl;
    if (swap_labels)
        std::cout << "WARNING: You interchanged the roles of the phenotypes."
                  << std::endl;
    #endif

    // used variables
    index_t num_paths = 0;                  // number of pathways
    index_t num_genes = 0;                  // number of unique gene symbols
    index_t num_type_A = 0;                 // number of patients class 0
    index_t num_type_B = 0;                 // number of patients class 1
    index_t num_patients = 0;               // number of overall patients

    std::vector<bitmp_t> opath;             // bitmap for each gene
    std::vector<exprs_t> exprs;             // expression data
    std::vector<label_t> labels;            // class label distribution
    std::vector<std::string> gsymb;         // list of gene symbol name
    std::vector<std::string> plist;         // list of patient names
    std::vector<std::string> pname;         // list of pathway names
    std::vector<std::vector<std::string>> pathw;

    // read expression data from gct file
    read_gct(gct_filename, exprs, plist, gsymb, num_genes, num_patients);

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_and_device_overall_exclusive_gct_parsing)
    #endif

    // transfer expression data to GPU
    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_to_device_transfer_expression_data)
    #endif
    exprs_t * Exprs = toGPU(exprs.data(), num_genes*num_patients);
    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_to_device_transfer_expression_data)
    #endif

    // transpose expression data (num_patiens x num_genes)
    transpose(Exprs, num_genes, num_patients);

    // read class label from cls file
    read_cls(cls_filename, labels, num_type_A, num_type_B);
    if (num_patients != num_type_A + num_type_B) {
        std::cout << "ERROR: Incompatible class file (cls) and expression "
                  << "data file, exiting." << std::endl;
        exit(CUDA_GSEA_INCOMPATIBLE_GCT_AND_CLS_FILE_ERROR);
    }

    // swap labels if demanded by user
    if (swap_labels) {
        for (auto& label : labels)
            label = (label+1) & 1;
        std::swap(num_type_A, num_type_B);
    }

    // transfer class labels to GPU
    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_to_device_transfer_labels)
    #endif
    label_t * Labels = toGPU(labels.data(), num_patients);
    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_to_device_transfer_labels)
    #endif

    // read pathways from gmt file
    read_gmt(gmt_filename, pname, pathw, num_paths);

    #ifdef CUDA_GSEA_PRINT_INFO
    print_gsea_info(num_patients, num_type_A, num_type_B, num_genes,
                    num_paths, num_perms, metric, gct_filename, cls_filename,
                    gmt_filename, sort_direction, swap_labels);
    #endif

    // create indices for genes
    unsigned int * OIndx = zeroGPU<unsigned int>(num_genes);
    range(OIndx, num_genes);

    // schedule computation
    index_t batch_size_perms = 0, num_batches_perms = 0,
            batch_size_paths = 0, num_batches_paths = 0, num_free_bytes = 0;
    schedule_computation<exprs_t, bitmp_t, enrch_t>
                        (num_genes, num_perms, num_paths,
                         batch_size_perms, num_batches_perms,
                         batch_size_paths, num_batches_paths, num_free_bytes);

    #ifdef CUDA_GSEA_PRINT_INFO
    print_scheduler_info(batch_size_perms, num_batches_perms,
                         batch_size_paths, num_batches_paths);
    #endif

    // vector for the storage of the enrichment scores
    std::vector<enrch_t> global_result(num_paths*num_perms);

    // for each batch of perms
    for (index_t batch_pi = 0; batch_pi < num_batches_perms; batch_pi++) {
        const index_t lower_pi = batch_pi*batch_size_perms;
        const index_t upper_pi = cumin(lower_pi+batch_size_perms,num_perms);
        const index_t width_pi = upper_pi-lower_pi;

        // initialize space for the correlation matrix and correlate
        exprs_t * Correl = zeroGPU<exprs_t>(width_pi*num_genes);
        correllate_genes_gpu(Exprs, Labels, Correl, num_genes, num_type_A,
                             num_type_B, width_pi, metric, lower_pi);

        unsigned int * Index = zeroGPU<unsigned int>(num_genes*width_pi);

        if (sort_direction)
            pathway_sort_gpu<exprs_t, unsigned int, index_t, false>
            (Correl, OIndx, Index, num_genes, width_pi);
        else
            pathway_sort_gpu<exprs_t, unsigned int, index_t, true>
            (Correl, OIndx, Index, num_genes, width_pi);

        // for each batch of paths
        for (index_t batch_pa = 0; batch_pa < num_batches_paths; batch_pa++) {
            const index_t lower_pa = batch_pa*batch_size_paths;
            const index_t upper_pa = cumin(lower_pa+batch_size_paths,num_paths);
            const index_t width_pa = upper_pa-lower_pa;

            // create bitmaps for pathways representation:
            create_bitmaps(gsymb, pathw, opath, lower_pa, width_pa);
            bitmp_t * OPath = toGPU(opath.data(), num_genes);

            #ifdef CUDA_GSEA_PRINT_INFO
            print_batch_info(lower_pa, upper_pa, width_pa,
                             lower_pi, upper_pi, width_pi);
            #endif

            // compute enrichment scores
            enrch_t * Score  = zeroGPU<enrch_t>(width_pa*width_pi);
            compute_scores_gpu(Correl, Index, OPath, Score,
                               num_genes, width_pi, width_pa);

            #ifdef CUDA_GSEA_PRINT_VERBOSE
            std::cout << "STATUS: free RAM on the GPU: "
                      << freeRAM()*1.0/(1<<30) << " GB." << std::endl;
            #endif

            // from now we can forget about the paths matrices
            cudaFree(OPath);                                              CUERR

            #ifdef CUDA_GSEA_PRINT_TIMINGS
            TIMERSTART(device_to_host_copy_scores);
            #endif

            // copy Score to vector for the storage of local results
            std::vector<enrch_t> local_result(width_pa*width_pi);
            cudaMemcpy(local_result.data(), Score,
                       sizeof(enrch_t)*width_pa*width_pi, D2H);           CUERR

            #ifdef CUDA_GSEA_PRINT_TIMINGS
            TIMERSTOP(device_to_host_copy_scores);
            #endif

            // copy local result to global result
            copy_strided(local_result.data(), global_result.data(),
                         num_paths, num_perms, lower_pa, upper_pa,
                         lower_pi, upper_pi);

            // get rid of the memory of the score matrix
            cudaFree(Score);                                              CUERR
        }

        // get rid of the memory
        cudaFree(Index);                                                  CUERR
        cudaFree(Correl);                                                 CUERR
    }

    // get rid of the memory
    cudaFree(OIndx);                                                      CUERR
    cudaFree(Exprs);                                                      CUERR
    cudaFree(Labels);                                                     CUERR

    #ifdef CUDA_GSEA_PRINT_INFO
    print_cuda_finished();
    #endif

    #ifdef CUDA_GSEA_PRINT_VERBOSE
    std::cout << "STATUS: free RAM on the GPU: "
              << freeRAM()*1.0/(1<<30) << " GB." << std::endl;
    #endif

    // dump data to disk if demanded
    if (dump_filename.size()) {
        dump_filename += "_"+std::to_string(num_paths);
        dump_filename += "_"+std::to_string(num_perms);
        dump_filename += "_"+std::to_string(8*sizeof(enrch_t));
        dump_filename += ".es";
        dump_binary(global_result.data(), num_paths*num_perms, dump_filename);
    }

    // write the result to command line
    auto result = final_statistics(global_result, num_paths, num_perms);
    for (index_t path = 0; path < num_paths; path++) {
        std::cout << std::showpos << std::fixed << "RESULT:"
                  << " ES: "   << result[path]
                  << " NES: "  << result[1*num_paths+path]
                  << " NP: "   << result[2*num_paths+path]
                  << " FWER: " << result[3*num_paths+path]
                  << "\t(" << pname[path] << ")" << std::endl;
    }

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_and_device_overall_exclusive_gct_parsing)
    #endif

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_and_device_overall)
    #endif
}

#endif
