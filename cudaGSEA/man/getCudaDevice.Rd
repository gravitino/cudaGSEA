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

\value{
An unsigned integer denoting the currently active CUDA device.
}

\examples{
# query active device
getCudaDevice()
}
