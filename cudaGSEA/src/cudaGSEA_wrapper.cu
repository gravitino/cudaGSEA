///////////////////////////////////////////////////////////////////////////////
// Greeter (if you remove this, then cite us at least.)
///////////////////////////////////////////////////////////////////////////////

// static void print_welcome() __attribute__((constructor));

///////////////////////////////////////////////////////////////////////////////
// Includes
///////////////////////////////////////////////////////////////////////////////

// C++ related
#include <iostream>                         // std::cout
#include <vector>                           // std::vector

// C related
#include <assert.h>                         // assert

// CUDA helpers
#include "include/cuda_helpers.cuh"         // convenience cmds, errors

// CUDA GSEA related stuff
#include "include/enrichment_scores.cuh"    // compute enrichments
#include "include/correlate_genes.cuh"      // correlate genes
#include "include/read_data_files.cuh"      // read gmt, gct and cls
#include "include/batch_scheduler.cuh"      // compute batches if ram too small
#include "include/create_bitmaps.cuh"       // pathway representation
#include "include/configuration.cuh"        // status messages and timings
#include "include/batched_sort.cuh"         // sort a bunch of arrays
#include "include/copy_strided.cuh"         // strided copy of results
#include "include/bitmap_types.cuh"         // bitmap types for pathways
#include "include/dump_binary.cuh"          // write result to disk
#include "include/print_info.cuh"           // information of the gsea routine
#include "include/statistics.cuh"           // final statistics for each path
#include "include/transpose.cuh"            // transposition of matrices

// R bindings
#include "cudaGSEA_wrapper.cuh"

///////////////////////////////////////////////////////////////////////////////
// Broad file format IO proxies
///////////////////////////////////////////////////////////////////////////////

void loadLabelsFromCLS_proxy(
        std::string cls_filename,
        std::vector<proxy_label_t>& labelList) {

    // dummy variables
    proxy_index_t num_type_A, num_type_B;
    read_cls(cls_filename, labelList, num_type_A, num_type_B);

    // sanity check
    assert(num_type_A+num_type_B == labelList.size());
}

void loadGeneSetsFromGMT_proxy(
        std::string gmt_filename,
        std::vector<proxy_strng_t>& pathwayNames,
        std::vector<std::vector<proxy_strng_t>>& pathwayList) {

    // dummy variables
    proxy_index_t num_paths;
    read_gmt(gmt_filename, pathwayNames, pathwayList, num_paths);

    // sanity check
    assert(pathwayNames.size() == num_paths);
}

void loadExpressionDataFromGCT_proxy(
        std::string gct_filename,
        std::vector<proxy_value_t>& exprsData,
        std::vector<proxy_strng_t>& geneSymbols,
        std::vector<proxy_strng_t>& patientList) {

    // dummy variables
    proxy_index_t num_genes, num_patients;
    read_gct(gct_filename, exprsData, patientList,
             geneSymbols, num_genes, num_patients);

    // sanity check
    assert(exprsData.size() == num_genes*num_patients);
}

///////////////////////////////////////////////////////////////////////////////
// CUDA settings / information proxies
///////////////////////////////////////////////////////////////////////////////

std::string create_device_string_helper(int id) {
    cudaDeviceProp property;
    cudaGetDeviceProperties(&property, id);                              CUERR

    auto name = property.name;
    auto size = property.totalGlobalMem;

    // construct the string
    std::string id_string = "device ";
    id_string += std::to_string(id);
    id_string += ": ";
    id_string += name;
    id_string += " with ";
    id_string += std::to_string((size+(1<<30)-1)/(1<<30));
    id_string += " GiB RAM";

    return id_string;
}

std::vector<std::string> listCudaDevices_proxy() {

    // the result vector
    std::vector<std::string> result;

    // get number of devices
    int num_devices;
    cudaGetDeviceCount(&num_devices);                                    CUERR

    // loop over devices and remember name
    for (int id = 0; id < num_devices; id++)
        result.push_back(create_device_string_helper(id));

    return result;
}

int getCudaDevice_proxy() {
    int id;
    cudaGetDevice(&id);                                                  CUERR
    return id;
}

int setCudaDevice_proxy(int id) {

    cudaSetDevice(id);
    auto active_id = getCudaDevice_proxy();

    if (active_id != id)
        std::cout << "WARNING: invalid device id, fallback to device "
                  << active_id << "." << std::endl;

    return active_id;
}

///////////////////////////////////////////////////////////////////////////////
// GSEA core methods
///////////////////////////////////////////////////////////////////////////////

template <class index_t>
void print_gsea_info_helper(index_t num_patients,
                            index_t num_type_A,
                            index_t num_type_B,
                            index_t num_genes,
                            index_t num_paths,
                            index_t num_perms,
                            std::string metric,
                            bool sort_direction,
                            bool swap_labels) {

    // a fancy bar suddenly appears in the wild:
    std::string bar = "===================";

    std::cout << "INFO: " << bar << bar << bar << std::endl;
    std::cout << "INFO: | The setting is the following:" << std::endl;
    std::cout << "INFO: | " << std::endl;
    std::cout << "INFO: | number of patients: " << num_patients << std::endl;
    std::cout << "INFO: | number of phenotype 0: " << num_type_A << std::endl;
    std::cout << "INFO: | number of phenotype 1: " << num_type_B << std::endl;
    std::cout << "INFO: | number of unique genes: " << num_genes << std::endl;
    std::cout << "INFO: | number of pathways: " << num_paths << std::endl;
    std::cout << "INFO: | number of permutations: " << num_perms << std::endl;
    std::cout << "INFO: | used ranking metric: " << metric << std::endl;
    std::cout << "INFO: " << bar << bar << bar << std::endl;

}

template <class exprs_t,                     // data type for expression data
          class index_t,                     // data type for indexing
          class label_t,                     // data type for storing labels
          class bitmp_t,                     // data type for storing pathways
          class enrch_t> __host__            // data type for enrichment scores
std::vector<enrch_t> gsea_proxy(std::vector<exprs_t>& exprs,
                                std::vector<std::string>& gsymb,
                                std::vector<std::string>& plist,
                                std::vector<label_t>& labels,
                                std::vector<std::string>& pname,
                                std::vector<std::vector<std::string>>& pathw,
                                std::string metric,
                                index_t num_perms,
                                bool sort_direction,
                                bool swap_labels,
                                std::string dump_filename) {

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
    index_t num_paths = pname.size();       // number of pathways
    index_t num_genes = gsymb.size();       // number of unique gene symbols
    index_t num_type_A = 0;                 // number of patients class 0
    index_t num_type_B = 0;                 // number of patients class 1
    index_t num_patients = plist.size();    // number of overall patients

    std::vector<bitmp_t> opath;             // bitmap for each gene

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

    // REMARK: DATA ALREADY TRANSPOSED
    // transpose expression data (num_patiens x num_genes)
    // transpose(Exprs, num_genes, num_patients);

    // get population count of phenotypes
    for (const auto& label : labels) {
        if (label == 0)
            num_type_A++;
        if (label == 1)
            num_type_B++;
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

    #ifdef CUDA_GSEA_PRINT_INFO
    print_gsea_info_helper(num_patients,
                           num_type_A,
                           num_type_B,
                           num_genes,
                           num_paths,
                           num_perms,
                           metric,
                           sort_direction,
                           swap_labels);
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

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_and_device_overall_exclusive_gct_parsing)
    #endif

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_and_device_overall)
    #endif

    return final_statistics(global_result, num_paths, num_perms);
}

std::vector<float> gsea_single_proxy(std::vector<float>& exprs,
                                     std::vector<std::string>& gsymb,
                                     std::vector<std::string>& plist,
                                     std::vector<char>& labels,
                                     std::vector<std::string>& pname,
                                     std::vector<std::vector<std::string>>& pathw,
                                     std::string metric,
                                     size_t num_perms,
                                     bool sort_direction,
                                     bool swap_labels,
                                     std::string dump_filename) {

    return gsea_proxy<float, size_t, char, bitmap32_t, float>
          (exprs, gsymb, plist, labels, pname, pathw,
           metric, num_perms, sort_direction, swap_labels, dump_filename);
}

std::vector<double> gsea_double_proxy(std::vector<double>& exprs,
                                      std::vector<std::string>& gsymb,
                                      std::vector<std::string>& plist,
                                      std::vector<char>& labels,
                                      std::vector<std::string>& pname,
                                      std::vector<std::vector<std::string>>& pathw,
                                      std::string metric,
                                      size_t num_perms,
                                      bool sort_direction,
                                      bool swap_labels,
                                      std::string dump_filename) {

    return gsea_proxy<double, size_t, char, bitmap64_t, double>
          (exprs, gsymb, plist, labels, pname, pathw,
           metric, num_perms, sort_direction, swap_labels, dump_filename);
}
