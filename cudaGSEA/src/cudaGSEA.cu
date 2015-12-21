#include "include/cugsea.cuh"                 // everything will work fine
#include "include/parse_cmd_line.cuh"         // parse command line options

void print_help() {
    auto errcode = system("cat HELP.txt");
}

int main (int argc, char * argv[]) {

    // Hello!
    print_welcome();

    if (argc == 1) {
        print_help();
        return 1;
    } else {

        // parse options
        auto is_res = cmd_option_exists(argv, argv+argc, "-res");
        auto is_cls = cmd_option_exists(argv, argv+argc, "-cls");
        auto is_gmx = cmd_option_exists(argv, argv+argc, "-gmx");
        auto is_num = cmd_option_exists(argv, argv+argc, "-nperm");
        auto is_met = cmd_option_exists(argv, argv+argc, "-metric");
        auto is_ord = cmd_option_exists(argv, argv+argc, "-order");
        auto is_dmp = cmd_option_exists(argv, argv+argc, "-dump");
        auto is_prc = cmd_option_exists(argv, argv+argc, "-precision");
        auto is_gpu = cmd_option_exists(argv, argv+argc, "-gpu");

        // check for mandatory parameters
        if (!is_res) {
            print_help();
            std::cout << "ERROR: You have to specify a collapsed gene "
                      << "expression file with -res. Exiting." << std::endl;
            exit(CUDA_GSEA_NO_GCT_FILE_SPECIFIED_ERROR);
        }

        if (!is_cls) {
            print_help();
            std::cout << "ERROR: You have to specify a class label assignment "
                      << "file with -cls. Exiting." << std::endl;
            exit(CUDA_GSEA_NO_GMT_FILE_SPECIFIED_ERROR);
        }

        // check mandatory parameters for validity
        auto gct_file = real_path(get_cmd_option(argv, argv + argc, "-res"));
        if (!gct_file.compare("")) {
            std::cout << "ERROR: File specified with -res could not be "
                      << "resolved by the operating system. Exiting."
                      << std::endl;
            exit(CUDA_GSEA_CANNOT_RESOLVE_GCT_FILE_ERROR);
        }

        auto cls_file = real_path(get_cmd_option(argv, argv + argc, "-cls"));
        if (!cls_file.compare("")) {
            std::cout << "ERROR: File specified with -cls could not be "
                      << "resolved by the operating system. Exiting."
                      << std::endl;
            exit(CUDA_GSEA_CANNOT_RESOLVE_CLS_FILE_ERROR);
        }

        auto gmt_file = real_path(get_cmd_option(argv, argv + argc, "-gmx"));
        if (!gmt_file.compare("")) {
            std::cout << "ERROR: File specified with -gmx could not be "
                      << "resolved by the operating system. Exiting."
                      << std::endl;
            exit(CUDA_GSEA_CANNOT_RESOLVE_GMT_FILE_ERROR);
        }

        // 16K permutations as default
        size_t nperm = 1<<14;
        if (is_num) {
            auto num = get_cmd_option(argv, argv + argc, "-nperm");
            if (check_number(num)) {
                nperm = to_number(num);
            } else {
                std::cout << "ERROR: number specified with -nperm could not "
                          << "be parsed as integer. Exiting." << std::endl;
                exit(CUDA_GSEA_CANNOT_PARSE_INTEGER_ERROR);
            }
        }

        // one pass signal2noise metric as default
        std::string metric = "onepass_signal2noise";
        if (is_met) {
            metric = get_cmd_option(argv, argv + argc, "-metric");
        }

        // descending sort direction as default
        bool sort_direction = true;
        if (is_ord) {
            auto order_predicate = get_cmd_option(argv, argv + argc, "-order");
            if (!order_predicate.compare("ascending"))
                sort_direction = false;
            else if (order_predicate.compare("descending")) {
                std::cout << "ERROR: option specified with -order must be "
                          << "either ascending or descending. Exiting."
                          << std::endl;
                exit(CUDA_GSEA_INVALID_ORDER_SPECIFIED_ERROR);
            }
        }

        // no file dump as default
        std::string dump_filename = "";
        if (is_dmp) {
            dump_filename = get_cmd_option(argv, argv + argc, "-dump");
            if (!dump_filename.compare("")) {
                std::cout << "ERROR: Parameter -dump observed but empty "
                          << "file name found. Exiting."
                          << std::endl;
                exit(CUDA_GSEA_EMPTY_DUMP_FILE_NAME_ERROR);
            }
        }

        // single precision as default
        bool single_precision = true;
        if (is_prc) {
            auto prec_string = get_cmd_option(argv, argv + argc, "-precision");
            if (!prec_string.compare("double"))
                single_precision = false;
            else if (prec_string.compare("single")) {
                std::cout << "ERROR: option specified with -precision must be "
                          << "either single or double. Exiting."
                          << std::endl;
                exit(CUDA_GSEA_INVALID_PRECISION_SPECIFIED_ERROR);
            }
        }

        // gpu 0 as default
        int device_id = 0;
        if (is_gpu) {
            auto identifier = get_cmd_option(argv, argv + argc, "-gpu");
            if (check_number(identifier)) {
                device_id = to_number(identifier);
            } else {
                std::cout << "ERROR: number specified with -gpu could not "
                          << "be parsed as integer. Exiting." << std::endl;
                exit(CUDA_GSEA_CANNOT_PARSE_INTEGER_ERROR);
            }
        }


        /*
        std::cout << gct_file << std::endl;
        std::cout << cls_file << std::endl;
        std::cout << gmt_file << std::endl;
        std::cout << nperm << std::endl;
        std::cout << metric << std::endl;
        std::cout << sort_direction << std::endl;
        std::cout << dump_filename << std::endl;
        std::cout << single_precision << std::endl;
        std::cout << device_id << std::endl;
        */

        // reset the CUDA device
        cudaSetDevice(device_id);
        cudaDeviceReset();

        TIMERSTART(overall)

        if (single_precision) {
            compute_gsea<float, size_t, unsigned char, bitmap32_t, float>
                        (gct_file, cls_file, gmt_file ,
                         metric, nperm, sort_direction, false, dump_filename);
        } else {
            compute_gsea<double, size_t, unsigned char, bitmap64_t, double>
                        (gct_file, cls_file, gmt_file ,
                         metric, nperm, sort_direction, false, dump_filename);
        }

        TIMERSTOP(overall)
    }
}


/*
int main (int argc, char * argv[]) {

    // Hello!
    print_welcome();


    // first of all we configure the used data types
    // see "include/configuration.cuh" to alter print settings
    typedef float exprs_t;         // data type for expression data
    typedef size_t index_t;        // data type for indexing
    typedef unsigned char label_t; // data type for storing labels, don't alter
    typedef bitmap64_t bitmp_t;    // data type for storing pathways
    typedef float enrch_t;         // data type for enrichment scores


    // reset the CUDA device
    cudaSetDevice(CUDA_DEVICE_ID);
    cudaDeviceReset();


    TIMERSTART(overall)
    // compute gene set enrichment for a given problem
    compute_gsea<exprs_t, index_t, label_t, bitmp_t, enrch_t> // data types
                ("../../../cugsea/data/GSE19429/GSE19429_series.gct",         // gct file
                 "../../../cugsea/data/GSE19429/GSE19429_series.cls",         // cls file
                 //"data/Pathways/c2.all.v5.0.symbols.gmt",     // gmt file
                 "../../../cugsea/data/Pathways/h.all.v5.0.symbols.gmt",     // gmt file
                 "onepass_signal2noise", 1<<14,
                 true, false, "/tmp/score");          // metric, perms
    TIMERSTOP(overall)

}
*/
