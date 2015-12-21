listCudaDevices <- function() {
    return (.Call("listCudaDevices"))
}

getCudaDevice <- function() {
    return (.Call("getCudaDevice"))
}

setCudaDevice <- function(deviceId) {
    return (.Call("setCudaDevice", deviceId))
}
