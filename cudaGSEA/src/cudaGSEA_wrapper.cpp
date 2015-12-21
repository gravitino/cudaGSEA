// c++
#include <iostream>
#include <limits>
#include <cmath>

// c
#include <Rcpp.h>

// custom
#include "cudaGSEA_wrapper.cuh"

///////////////////////////////////////////////////////////////////////////////
// Broad file format IO
///////////////////////////////////////////////////////////////////////////////

RcppExport SEXP loadLabelsFromCLS(SEXP clsFilename) {

    auto cls_filename = Rcpp::as<proxy_strng_t>(clsFilename);
    std::vector<proxy_label_t> labelList;
    loadLabelsFromCLS_proxy(cls_filename, labelList);

    return Rcpp::wrap(labelList);
}

RcppExport SEXP loadGeneSetsFromGMT(SEXP gmtFileName) {

    auto gmt_filename = Rcpp::as<proxy_strng_t>(gmtFileName);
    std::vector<proxy_strng_t> pathwayNames;
    std::vector<std::vector<proxy_strng_t>> pathwayList;

    loadGeneSetsFromGMT_proxy(gmt_filename, pathwayNames, pathwayList);
    auto result = Rcpp::List(pathwayList.begin(), pathwayList.end());
    result.attr("names") = pathwayNames;

    return result;
}

RcppExport SEXP loadExpressionDataFromGCT(SEXP gctFileName) {

    auto gct_filename = Rcpp::as<proxy_strng_t>(gctFileName);
    std::vector<proxy_value_t> exprsData;
    std::vector<proxy_strng_t> geneSymbols;
    std::vector<proxy_strng_t> patientList;

    loadExpressionDataFromGCT_proxy(gct_filename, exprsData,
                                    geneSymbols, patientList);

    const proxy_index_t height = geneSymbols.size();
    const proxy_index_t width  = patientList.size();
    auto matrix = Rcpp::NumericMatrix(Rcpp::Dimension(height, width));

    // transpose expression data due to R's col-major-order indexing
    for (proxy_index_t i = 0; i < height; i++)
        for (proxy_index_t j = 0; j < width; j++)
            matrix[j*height+i] = exprsData[i*width+j];

    // attach the dimension labels
    std::vector<std::vector<proxy_strng_t>> dimnames;
    dimnames.push_back(std::move(geneSymbols));
    dimnames.push_back(std::move(patientList));
    matrix.attr("dimnames") = dimnames;

    return Rcpp::wrap(Rcpp::Language("as.data.frame", matrix).eval());
}

///////////////////////////////////////////////////////////////////////////////
// CUDA settings / information
///////////////////////////////////////////////////////////////////////////////

RcppExport SEXP listCudaDevices() {
    return Rcpp::wrap(listCudaDevices_proxy());
}

RcppExport SEXP getCudaDevice() {
    return Rcpp::wrap(getCudaDevice_proxy());
}

RcppExport SEXP setCudaDevice(SEXP deviceId) {
    return Rcpp::wrap(setCudaDevice_proxy(Rcpp::as<int>(deviceId)));
}

///////////////////////////////////////////////////////////////////////////////
// GSEA core
///////////////////////////////////////////////////////////////////////////////

RcppExport SEXP GSEA(SEXP exprsData,
                     SEXP labelList,
                     SEXP geneSets,
                     SEXP numPermutations,
                     SEXP metricString,
                     SEXP dumpFilename,
                     SEXP checkInput,
                     SEXP doublePrecision) {

    try {

        // get the data frame of expression values
        auto data_frame = Rcpp::DataFrame(exprsData);

        // extract row and column names
        Rcpp::CharacterVector colnames_ = data_frame.attr("names");
        Rcpp::CharacterVector rownames_ = data_frame.attr("row.names");
        auto colnames = Rcpp::as<std::vector<proxy_strng_t>>(colnames_);
        auto rownames = Rcpp::as<std::vector<proxy_strng_t>>(rownames_);

        // from now we cast to matrix
        auto cast_funct = Rcpp::Language("as.matrix", data_frame);
        Rcpp::NumericMatrix exprs_table = cast_funct.eval();
        Rcpp::Dimension dimension = exprs_table.attr("dim");

        // get the gene sets
        typedef std::vector<std::vector<proxy_strng_t>> gene_set_t;
        auto gene_sets = Rcpp::as<gene_set_t>(geneSets);
        auto num_paths = gene_sets.size();

        // TODO: here we have to extract names from the gene sets
        auto gene_list = Rcpp::List(geneSets);
        auto pnames = Rcpp::as<std::vector<std::string>>(gene_list.attr("names"));

        // if people forget to name the gene sets
        for (proxy_index_t i = 0; i < pnames.size(); i++)
            if (pnames[i] == "NA") {
                pnames.resize(num_paths);
                for (proxy_index_t j = 0; j < num_paths; j++)
                    pnames[j] = std::to_string(j+1);
                std::cout << "WARNING: We enumerated your gene set names "
                          << "from 1 to n." << std::endl;
                break;
            }


        // get the labels
        auto label_vector = Rcpp::as<std::vector<proxy_label_t>>(labelList);

        // deep copy labels to char since Rcpp can not accomplish it
        std::vector<char> labels;
        for (const auto& label : label_vector)
            labels.push_back(label);


        // get number of permutations
        auto num_permutations = Rcpp::as<proxy_index_t>(numPermutations);

        // get metric string
        auto metric_string = Rcpp::as<proxy_strng_t>(metricString);

        // get filename where to dump all enrichment scores
        auto dump_filename = Rcpp::as<proxy_strng_t>(dumpFilename);

        // get check input boolean
        auto check_input = Rcpp::as<bool>(checkInput);

        // get double precision boolean
        auto double_precision = Rcpp::as<bool>(doublePrecision);

        if (check_input) {
            // check for compatible dimensions between data frame and matrix
            if (dimension[0] != rownames.size()) {
                std::cout << "INVALID DATA: the vertical dimension and the "
                          << "number of row names do not comply." << std::endl;
                return Rcpp::wrap(false);
            }

            if (dimension[1] != colnames.size()) {
                std::cout << "INVALID DATA: the horizontal dimension and the "
                          << "number of column names do not comply."
                          << std::endl;
                return Rcpp::wrap(false);
            }

            // check for compatible label vector size
            if (label_vector.size() != dimension[1]) {
                std::cout << "INVALID DATA: Your label list has a different "
                          << "amount of entries as the number of columns in "
                          << "your gene  expression data frame." << std::endl;

                return Rcpp::wrap(false);
            }

            // check for binary label distribution
            for (const auto& label : label_vector)
                if (label != 0 && label != 1) {
                    std::cout << "INVALID DATA: Label vector may only contain "
                              << "zeros and ones since cudaGSEA only supports "
                              << "two classes of phenotypes." << std::endl;
                    return Rcpp::wrap(false);
                }

            // check for extrema and NaNs
            auto maximum = -std::numeric_limits<proxy_value_t>::infinity();
            auto minimum = +std::numeric_limits<proxy_value_t>::infinity();

            for (const auto& value : exprs_table) {
                if(isnan(value)) {
                    std::cout << "INVALID DATA: expression table has NaN "
                              << "entries that cannot be handled by the GSEA "
                              << "algorithm." << std::endl;
                    return Rcpp::wrap(false);
                } else {
                    maximum = value > maximum ? value : maximum;
                    minimum = value < minimum ? value : minimum;
                }
            }

            // warning when negative values in the expression table
            if (minimum < 0)
                std::cout << "WARNING: There are negative expression values "
                          << "in the provided table. Some metrics cannot "
                          << "handle negative amplitudes e.g. the "
                          << "log2_ratio_of_classes family." << std::endl;

            // check for empty gene set list
            if (gene_sets.size() == 0) {
                std::cout << "INVALID DATA: you must at least specify one "
                          << "gene set." << std::endl;
                return Rcpp::wrap(false);
            }

            // check for empty gene sets
            for (const auto& gene_set : gene_sets)
                if (gene_set.size() == 0) {
                    std::cout << "INVALID DATA: There is an empty gene set "
                              << "(no gene identifiers) in your list of "
                              << "gene sets." << std::endl;
                    return Rcpp::wrap(false);
                }
        }

        if (double_precision) {

            auto exprs = Rcpp::as<std::vector<double>>(exprs_table);
            auto result = gsea_double_proxy(exprs, rownames, colnames, labels, pnames, gene_sets, metric_string, num_permutations, true, false, "");

            Rcpp::NumericVector   ES(result.begin()+0*num_paths,
                                     result.begin()+1*num_paths);
            Rcpp::NumericVector  NES(result.begin()+1*num_paths,
                                     result.begin()+2*num_paths);
            Rcpp::NumericVector   NP(result.begin()+2*num_paths,
                                     result.begin()+3*num_paths);
            Rcpp::NumericVector FWER(result.begin()+3*num_paths,
                                     result.begin()+4*num_paths);

            auto frame = Rcpp::DataFrame::create(Rcpp::_["ES"]   = ES,
                                                 Rcpp::_["NES"]  = NES,
                                                 Rcpp::_["NP"]   = NP,
                                                 Rcpp::_["FWER"] = FWER);
            frame.attr("row.names") = Rcpp::CharacterVector(pnames.begin(),
                                                            pnames.end());
            return frame;

        } else {

            auto exprs = Rcpp::as<std::vector<float>>(exprs_table);
            auto result = gsea_single_proxy(exprs, rownames, colnames, labels, pnames, gene_sets, metric_string, num_permutations, true, false, "");

            Rcpp::NumericVector   ES(result.begin()+0*num_paths,
                                     result.begin()+1*num_paths);
            Rcpp::NumericVector  NES(result.begin()+1*num_paths,
                                     result.begin()+2*num_paths);
            Rcpp::NumericVector   NP(result.begin()+2*num_paths,
                                     result.begin()+3*num_paths);
            Rcpp::NumericVector FWER(result.begin()+3*num_paths,
                                     result.begin()+4*num_paths);

            auto frame = Rcpp::DataFrame::create(Rcpp::_["ES"]   = ES,
                                                 Rcpp::_["NES"]  = NES,
                                                 Rcpp::_["NP"]   = NP,
                                                 Rcpp::_["FWER"] = FWER);
            frame.attr("row.names") = Rcpp::CharacterVector(pnames.begin(),
                                                            pnames.end());
            return frame;
        }

    } catch (Rcpp::not_compatible x) {
        std::cout << "INVALID DATA: cannot cast data. Make sure you "
                  << "fill in the correct datastructures." << std::endl;

        return Rcpp::wrap(false);
    }
}
