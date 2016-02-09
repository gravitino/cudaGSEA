\name{listCudaDevices}
\alias{listCudaDevices}
\title{Query system for CUDA-enabled devices}

\description{
This method queries the system for CUDA-enabled GPUs and returns
a list of identifier strings together with the attached RAM.
}

\usage{
listCudaDevices()
}

\value{
A list of strings identifying the CUDA devices by name and amount of RAM.
}

\examples{
    # list all devices
    listCudaDevices()
}
