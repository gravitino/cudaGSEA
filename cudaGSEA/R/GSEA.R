GSEA <- function(exprsData, labelList, geneSets, numPermutations, 
    metricString="", dumpFileName="", checkInput=TRUE, doublePrecision=FALSE) {
    return (.Call("GSEA", exprsData, labelList, geneSets, numPermutations,
        metricString, dumpFileName, checkInput, doublePrecision))
}
