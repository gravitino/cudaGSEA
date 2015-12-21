\name{loadGeneSetsFromGMT}
\alias{loadGeneSetsFromGMT}
\title{Set the active CUDA device}

\description{
This method loads gene sets from *.gmt files as specified by the Broad
Institute of MIT and Harvard:

http://www.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats 
}

\usage{
loadGeneSetsFromGMT(gmtFileName)
}

\arguments{
\item{gmtFileName}{Specifies the path to a *.gmt file with a string.} 
}

\examples{
    # geneSets <- loadGeneSetsFromGMT("/path/to/gene/sets/file.gmt")
    print("uncommend the line above if you have a gmt file!")
}
