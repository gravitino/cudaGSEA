\name{setCudaDevice}
\alias{setCudaDevice}
\title{Set the active CUDA device}

\description{
	This method sets the active CUDA device -- a direct wrapper
        of CUDA's cudaSetDevice method.
}

\usage{
setCudaDevice(deviceId)
}

\arguments{
\item{deviceId}{An unsigned integer that specifies the device ID.} 
}

\examples{
    # set device to 0
    setCudaDevice(0)
}
