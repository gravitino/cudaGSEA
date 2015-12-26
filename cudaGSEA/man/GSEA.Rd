\name{GSEA}
\alias{GSEA}
\title{Perform a Gene Set Enrichment Analysis}

\description{
This methods performs a Gene Set Enrichment Analysis according to the
method proposed by Subramanian et al. "Gene set enrichment analysis: 
A knowledge-based approach for interpreting genome-wide expression 
profiles" using CUDA-enabled accelerators.
}

\usage{
GSEA(exprsData, labelList, geneSets, numPermutations, metricString, 
    dumpFileName, checkInput, doublePrecision)
}

\arguments{
\item{exprsData}{Data frame where rows enumerate gene symbols and columns
    identify patient names. The latter will not be used by 
    the GSEA algorithm. Use loadExpressionDataFromGCT in order
    to obtain an example.
}
\item{labelList}{Binary vector consisting of zeros and ones that indicates
    the affiliation of each patient to one of the two phenotypes. 
    Use loadLabelsFromCLS in order to obtain an example.
}
\item{geneSets}{A named list of gene sets each consisting of a vector of gene 
    symbols. Use loadGeneSetsFromGMT in order to obtain an example.
}
\item{numPermutations}{Unsigned integer that specifies the number of 
    permutations used during the permutation test.
}
\item{metricString}{String that specifies the used local metric. Choose from

=== one-pass mean rankings (fastest) ===
(computed with the following formula Mu(X) = E(X))

- naive_diff_of_classes
- naive_ratio_of_classes
- naive_log2_ratio_of_classes

=== one-pass Kahan-stable mean rankings (marginally slower, more precise) ===
(see  Kahan, William (January 1965), "Further remarks on reducing truncation 
errors", Communications of the ACM 8 (1): 40, doi:10.1145/363707.363723)

- stable_diff_of_classes
- stable_ratio_of_classes
- stable_log2_ratio_of_classes

=== one-pass Steiner-naive mean and stdev rankings (fastest, few patients) ===
(computed with the following formula Var(X) = E(X^2)-E(X)^2)

- onepass_signal2noise
- onepass_t_test

=== two-pass mean and stdev ranking (slower, more precise, best trade-off) ===
(computed with the following formula Var(X) = E(X-E(X))^2)

- twopass_signal2noise
- twopass_t_test

=== one-pass Knuth-stable mean and stdev rankings (slower, more precise) ===
(see Donald E. Knuth (1998). The Art of Computer Programming, volume 2: 
Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.)

- stable_signal2noise
- stable_t_test

=== two-pass Kahan-stable mean and Bjoerck-compensated stdev (slowest) ===
(see http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)

- overkill_signal2noise
- overkill_t_test
}

\item{dumpFileName}{String that specifies the full path to binary dump of 
    either "float" or "double" floating point values. This
    argument is optional. An empty string configures GSEA
    to not dump the full enrichment score table.
}
\item{checkInput}{Boolean that specifies if either the input of GSEA is
    checked for sanity or not. This argument is optional. 
    Leave this true (default) unless you really know what 
    you are doing!
}
\item{doublePrecision}{Boolean that specifies if either single precision is 
    used (false) or double precision (true). This argument
    is optional (default: false) Please note, double
    precision may be executed on consumer GPUs such
    as the Geforce Titan X. However, Tesla GPUs usually 
    perform reasonably better in means of runtime.
}
}

\value{
A data frame holding local and global statistics for each gene set.
}

\examples{

# read data from cls, gmt and gct file formats
# exprsData <- loadExpressionDataFromGCT("data/GSE19429/GSE19429_series.gct")
# labelList <- loadLabelsFromCLS("data/GSE19429/GSE19429_series.cls")
# geneSets <- loadGeneSetsFromGMT("data/Pathways/h.all.v5.0.symbols.gmt")

# access CUDA devices
listCudaDevices()
setCudaDevice(0)

# configure GSEA
nperm <- 1024                        # number of permutations
metric <- "onepass_signal2noise"     # metric string see README.md
dump <- ""                           # if not empty path to binary dump
checkInput <-TRUE                    # check first three inputs or not
doublePrecision <-FALSE              # compute in single or double prec.

# GSEA(exprsData, labelList, geneSets, nperm, metric , 
#      dump, checkInput, doublePrecision)
}
