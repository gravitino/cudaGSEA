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

\examples{
    # list all devices
    listCudaDevices()
}
