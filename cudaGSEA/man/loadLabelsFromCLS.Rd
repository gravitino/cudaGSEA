\name{loadLabelsFromCLS}
\alias{loadLabelsFromCLS}
\title{Load a binary label vector from *.cls file format.}

\description{
This method loads a binary label vector encoding the phenotypes with
zeros and ones from a *.cls file as defined by the Broad Institute
of MIT and Harvard:

http://www.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats
}

\usage{
loadLabelsFromCLS(clsFileName)
}

\arguments{
\item{clsFileName}{Specifies the path to a *.cls file with a string.} 
}

\value{
A binary vector denoting the phenotype affiliation.
}

\examples{
# labels <- loadLabelsFromCLS("/path/to/labels/file.cls")
print("uncommend the line above if you have a cls file!")
}
