loadLabelsFromCLS <- function(clsFileName) {
    return (.Call("loadLabelsFromCLS", clsFileName))
}

loadGeneSetsFromGMT <- function(gmtFileName) {
    return(.Call("loadGeneSetsFromGMT", gmtFileName))
}

loadExpressionDataFromGCT <- function(gctFileName) {
    return(.Call("loadExpressionDataFromGCT", gctFileName))
}
