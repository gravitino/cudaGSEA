library(cudaGSEA)                            # the cudaGSEA library

# read data from cls, gmt and gct file formats
exprsData <- loadExpressionDataFromGCT("data/GSE19429/GSE19429_series.gct")
labelList <- loadLabelsFromCLS("data/GSE19429/GSE19429_series.cls")
geneSets <- loadGeneSetsFromGMT("data/Pathways/h.all.v5.0.symbols.gmt")

# access CUDA devices
listCudaDevices()
setCudaDevice(0)
getCudaDevice()

# configure GSEA
nperm <- 1024                                # number of permutations
metric <- "onepass_signal2noise"             # metric string see README.md
dump <- ""                                   # if not empty path to binary dump
checkInput <-TRUE                            # check first three inputs or not
doublePrecision <-FALSE                      # compute in single or double prec.

GSEA(exprsData, labelList, geneSets, nperm, metric , dump, checkInput, doublePrecision)
