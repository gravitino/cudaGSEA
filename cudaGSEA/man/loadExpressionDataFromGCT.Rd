\name{loadExpressionDataFromGCT}
\alias{loadExpressionDataFromGCT}
\title{Load an expression data matrix from *.gct file.}

\description{
This method loads a gene expression data matrix from a *.gct file
as specified by the Broad Institute of MIT and Harvard:

http://www.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats
}

\usage{
loadExpressionDataFromGCT(gctFileName)
}

\arguments{
\item{gctFileName}{Specifies the path to a *.gct file with a string.} 
}

\examples{
# expressionData <- loadExpressionDataFromGCT("/path/to/expressions/file.gct")
print("uncommend the line above if you have a gct file!")
}
