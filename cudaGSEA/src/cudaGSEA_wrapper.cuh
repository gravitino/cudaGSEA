#include <iostream>
#include <vector>
#include <string>

// fixed data types for the communication with R
typedef std::string  proxy_strng_t;
typedef double       proxy_value_t;
typedef size_t       proxy_label_t;
typedef size_t       proxy_index_t;

///////////////////////////////////////////////////////////////////////////////
// Broad file format IO
///////////////////////////////////////////////////////////////////////////////

void loadLabelsFromCLS_proxy(std::string,
                             std::vector<proxy_label_t>&);

void loadGeneSetsFromGMT_proxy(std::string,
                               std::vector<proxy_strng_t>&,
                               std::vector<std::vector<proxy_strng_t>>&);

void loadExpressionDataFromGCT_proxy(std::string,
                                     std::vector<proxy_value_t>&,
                                     std::vector<proxy_strng_t>&,
                                     std::vector<proxy_strng_t>&);

///////////////////////////////////////////////////////////////////////////////
// CUDA settings / information proxies
///////////////////////////////////////////////////////////////////////////////

std::vector<proxy_strng_t> listCudaDevices_proxy();
int getCudaDevice_proxy();
int setCudaDevice_proxy(int);

///////////////////////////////////////////////////////////////////////////////
//  GSEA methods
///////////////////////////////////////////////////////////////////////////////

std::vector<float> gsea_single_proxy(std::vector<float>&,
                                     std::vector<std::string>&,
                                     std::vector<std::string>&,
                                     std::vector<char>&,
                                     std::vector<std::string>&,
                                     std::vector<std::vector<std::string>>&,
                                     std::string,
                                     size_t,
                                     bool,
                                     bool,
                                     std::string);

std::vector<double> gsea_double_proxy(std::vector<double>&,
                                      std::vector<std::string>&,
                                      std::vector<std::string>&,
                                      std::vector<char>&,
                                      std::vector<std::string>&,
                                      std::vector<std::vector<std::string>>&,
                                      std::string,
                                      size_t,
                                      bool,
                                      bool,
                                      std::string);
