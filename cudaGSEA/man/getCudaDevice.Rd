\name{GetCudaDevice}
\alias{getCudaDevice}
\title{Get the active CUDA device}

\description{
This method gets the active CUDA device -- a direct wrapper
of CUDA's cudaGetDevice method.
}

\usage{
getCudaDevice()
}


\examples{
# query active device
getCudaDevice()
}
