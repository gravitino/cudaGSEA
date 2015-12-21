#ifndef CUDA_GSEA_READ_DATA_FILES_CUH
#define CUDA_GSEA_READ_DATA_FILES_CUH

#include <unordered_set>
#include <functional> 
#include <algorithm> 
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cctype>
#include <locale>

#include "cuda_helpers.cuh"
#include "error_codes.cuh"
#include "configuration.cuh"

///////////////////////////////////////////////////////////////////////////////
// trimming gene symbol names (inspired by stack overflow)
///////////////////////////////////////////////////////////////////////////////

// trim from start
static inline std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), 
                std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), 
                std::not1(std::ptr_fun<int, int>(std::isspace))).base(), 
                s.end());
        return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
}

///////////////////////////////////////////////////////////////////////////////
// read expression data from gct file
///////////////////////////////////////////////////////////////////////////////

template <class exprs_t, class index_t> __host__
void read_gct(const std::string filename, 
              std::vector<exprs_t>& exprs,       // expression table
              std::vector<std::string>& plist,   // patient list
              std::vector<std::string>& gsymb,   // gene symbols
              index_t& num_genes,                // number of genes
              index_t& num_patients) {           // number of patients

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_read_gct_file);
    #endif

    std::ifstream gctfile (filename, std::ios::in);
    std::unordered_set<std::string> gene_symbol_set;

    if (gctfile.is_open()) {
        
        std::string token;

        // check header
        std::getline(gctfile, token);
        if (token != "#1.2") {
            std::cout << "Error: First line of " << filename
                      << " is not a valid gct header, exiting." << std::endl;
            exit(CUDA_GSEA_INCORRECT_GCT_HEADER_ERROR);
        }

        // extract dimensions and prepare data structures
        gctfile >> num_genes >> num_patients;
        exprs.resize(num_genes*num_patients);
        plist.resize(num_patients);
        gsymb.resize(num_genes);

        // populate plist
        std::getline(gctfile, token, '\t');               // NAME
        std::getline(gctfile, token, '\t');               // DESCRIPTION
        for (index_t patient = 0; patient < num_patients-1; patient++)
            std::getline(gctfile, plist[patient], '\t');
        std::getline(gctfile, plist[num_patients-1], '\n');

        // parse the remaining file
        for (index_t gene = 0; gene < num_genes; gene++) {

            // get gene symbol
            std::getline(gctfile, token, '\t');           // gene symbol
            gsymb[gene] = trim(token);                    // remove whitespaces
            std::getline(gctfile, token, '\t');           // ignore description

            // check for duplicates
            if(gene_symbol_set.find(gsymb[gene]) == gene_symbol_set.end()) {
                gene_symbol_set.insert(gsymb[gene]);
            } else {
                std::cout << "Error: duplicate symbol " << gsymb[gene]
                          << " , exiting." << std::endl;
                exit(CUDA_GSEA_DUPLICATE_GENE_SYMBOL_ERROR);
            }


            // get expression values
            for (index_t patient = 0; patient < num_patients; patient++)
                gctfile >> exprs[gene*num_patients+patient];
        }

        gctfile.close();

    } else {
        std::cout << "Error: Unable to open file " << filename 
                  << " , exiting." << std::endl;
        exit(CUDA_GSEA_CANNOT_OPEN_FILE_ERROR);
    }

        #ifdef CUDA_GSEA_PRINT_VERBOSE
        std::cout << "STATUS: In " << filename << " found " 
                  << num_patients << " patients" << std::endl
                  << "STATUS: and " << num_genes << " unqiue gene symbols." 
                  << std::endl;
        #endif

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_read_gct_file);
    #endif
}

///////////////////////////////////////////////////////////////////////////////
// read pathways from gmt file
///////////////////////////////////////////////////////////////////////////////

template <class index_t> __host__ 
void read_gmt(const std::string filename,
                    std::vector<std::string>& pname,
                    std::vector<std::vector<std::string>>& pathw,
                    index_t& num_paths) {

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_read_gmt_file);
    #endif

    std::ifstream gmtfile (filename, std::ios::in);

    if (gmtfile.is_open()) {

        // reset data structures
        pname.resize(0);
        pathw.resize(0);
        std::string line;
        std::string token; 

        while(std::getline(gmtfile, line)) {
    
            std::stringstream ss(line);
            std::vector<std::string> symbols;

            // parse pathway name
            std::getline(ss, token, '\t');
            pname.push_back(token);

            // parse symbols
            size_t count = 0;
            while (std::getline(ss, token, '\t'))
                if (count++)
                    symbols.push_back(trim(token));

            pathw.push_back(std::move(symbols));
        }

        num_paths = pathw.size();

        #ifdef CUDA_GSEA_PRINT_VERBOSE
        std::cout << "STATUS: In " << filename << " found " << pname.size()
                  << " pathways." << std::endl;
        #endif

    } else {
        std::cout << "Error: Unable to open file " << filename 
                  << " , exiting." << std::endl;
        exit(CUDA_GSEA_CANNOT_OPEN_FILE_ERROR);
    }

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_read_gmt_file);
    #endif
}

///////////////////////////////////////////////////////////////////////////////
// read labels from cls file REMARK: no support for continuous #numeric labels
///////////////////////////////////////////////////////////////////////////////

template <class label_t, class index_t> __host__
void read_cls(const std::string filename,
                    std::vector<label_t>& labels,
                    index_t& num_type_A,
                    index_t& num_type_B) {

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTART(host_read_cls_file);
    #endif

    std::ifstream clsfile (filename, std::ios::in);

    if (clsfile.is_open()) {

        // get the first line
        std::string line;
        std::getline(clsfile, line);
        std::stringstream ss(line);

        // parse header
        index_t num_patients = 0, num_classes = 0, dummy = 0;
        ss >> num_patients >> num_classes >> dummy;

        if (num_classes != 2) {
            std::cout << "ERROR: Number of phenotype classes specified in the"
                      << " corresponding cls file " << filename
                      << " must be exactly two, exiting." << std::endl;
            exit(CUDA_GSEA_UNSUPPORTED_NUMBER_OF_PHENOTYPES_ERROR);
        }

        if (dummy != 1) {
            std::cout << "ERROR: Invalid header found in the corresponding"
                      << "cls file " << filename << " , exiting." << std::endl;
            exit(CUDA_GSEA_INCORRECT_CLS_HEADER_ERROR);
        }

        // resize label vector to fit the number of patients
        labels.resize(num_patients);

        // get the second line
        std::getline(clsfile, line);
        std::stringstream ss2(line);

        // ingredients of second line
        std::string hashtag;
        std::string class_one = "";
        std::string class_two = "";

        // parse them (remove hashtag and whitespaces, then parse)
        std::getline(ss2, hashtag, '#');
        while(!class_one.size())
            std::getline(ss2, class_one, ' ');
        std::getline(ss2, class_two, '\n');
        class_one=trim(class_one);
        class_two=trim(class_two);

        // get the third line
        std::getline(clsfile, line);
        std::stringstream ss3(line);

        // parse the symbolic labels
        std::vector<std::string> symbolic_labels(num_patients);
        for (index_t id = 0; id < num_patients; id++)
            ss3 >> symbolic_labels[id];

        // determine the corresponding class labels
        index_t id = 0;
        for (const auto& x : symbolic_labels) {

            // final class label
            label_t value;

            // This might not find all pathological errors in cls files
            // but should be sufficient to exclude obvious errors. Here class 
            // labels are a assigned using the symbolic class labels unless 
            // no matching symbol is found but otherwise represented by "0" 
            // or "1". However, we do not check for the mixing of "0"/"1" and 
            // symbolic phenotype labels. REMARK: @Broad Institute: Who defines 
            // such an ill-defined format?
            if (x == class_one) 
                value = 0;
            else if (x == class_two)
                value = 1;
            else if (x == "0")
                value = 0;
            else if (x == "1")
                value = 1;
            else {
                std::cout << "ERROR: Unknown label in cls file " << filename
                          << " : \"" << x << "\", exiting." << std::endl;
                exit(CUDA_GSEA_UNKNOWN_LABEL_IN_CLS_FILE_ERROR);
            }

            // write the final label
            labels[id++] = value;
        }

        // determine the number of patients in phenotype A and B
        num_type_A = 0; num_type_B = 0;
        for (const auto& x : labels) {
            if (x == 0)
                num_type_A++;
            if (x == 1)
                num_type_B++;
        }

        // last sanity check
        if (num_type_A + num_type_B != num_patients) {
            std::cout << "ERROR: Number of phenotypes in both classes do not"
                      << " add up to the number of patients in file "
                      << filename << " , exiting." << std::endl;
            exit(CUDA_GSEA_NON_SPECIFIED_CORRUPTION_IN_CLS_FILE_ERROR);
        }

    } else {
        std::cout << "Error: Unable to open file " << filename 
                  << " , exiting." << std::endl;
        exit(CUDA_GSEA_CANNOT_OPEN_FILE_ERROR);
    }

        #ifdef CUDA_GSEA_PRINT_VERBOSE
        std::cout << "STATUS: In " << filename << " found " 
                  << (num_type_A+num_type_B) << " labels." << std::endl;
        #endif

    #ifdef CUDA_GSEA_PRINT_TIMINGS
    TIMERSTOP(host_read_cls_file);
    #endif
}


#endif
