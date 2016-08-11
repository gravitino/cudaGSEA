import numpy as np
import array as ar
import pylab as pl
import gzip
import matplotlib.gridspec as gridspec

try:
    import seaborn
    seaborn.set_style("whitegrid")
except:
    pass

def get_geneset_names_from_gmt(filename):

    gensets = []

    with gzip.open(filename, "r") as f:
        for line in f:
            gensets.append(line.split()[0])

    return gensets

def get_data_from_es(filename):

    num_perms, num_paths = 0, 0
    tokens = filename.split(".")[-3].split("_")

    print "Opening file:", filename
    with gzip.open(filename, "rb") as f:
        if tokens[-1] == "64":
            data = np.array(ar.array("d", f.read()))
        elif tokens[-1] == "32":
            data = np.array(ar.array("f", f.read()))
        else:
            print "ERROR: Incompatible file name, expecting 32 or 64 at the end"
            return
        try:
            num_perms = int(tokens[-2])
            num_paths = int(tokens[-3])
        except:
            print "ERROR: Cannot parse dimension of the matrix"

    print num_perms, num_paths
    return data.reshape((num_paths, num_perms))

def get_data_from_edb(filename):

    lines = []
    result = {}
    enrichment_scores={}

    with gzip.open(filename, "r") as f:
        for line in f:
            if "DTG RANKED_LIST=" in line:
                lines.append(line)

    for index, line in enumerate(lines):
        line = line.split('"')
        name = line[5].split("#")[-1]
        result[name] = np.array(map(float, line[17].split()))
        enrichment_scores[name] = float(line[7])

    return result, enrichment_scores


cudaGSEA = get_data_from_es ("cudaGSEA_seed42_hallmark.50_1048576_64.es.gz").round(4)
broadGSEA0, scores0 = get_data_from_edb("broadGSEA_seed149_hallmark.50_1048576_64.edb.gz")
broadGSEA1, scores1 = get_data_from_edb("broadGSEA_seed150_hallmark.50_1048576_64.edb.gz")
genesets = get_geneset_names_from_gmt("h.all.v5.0.symbols.gmt.gz")


print cudaGSEA.shape, len(broadGSEA0), len(broadGSEA1)

bins = np.linspace(-1, 1, 1024)
dom  = np.linspace(-1, 1, len(bins)-1)

bc, bb = [], []

for cuda, name in zip(cudaGSEA, genesets):

    print cuda[0], scores0[name], scores1[name] 

    cuda = np.histogram(cuda, bins=bins, density=True)
    broad0 = np.histogram(broadGSEA0[name], bins=bins, density=True)
    broad1 = np.histogram(broadGSEA1[name], bins=bins, density=True)

    
    Cuda = np.cumsum(cuda[0])
    Cuda /= Cuda[-1]

    Broad0 = np.cumsum(broad0[0])
    Broad0 /= Broad0[-1]
    
    Broad1 = np.cumsum(broad1[0])
    Broad1 /= Broad1[-1]

    ax=gridspec.GridSpec(1, 1)
    ax.update(left=0.1, right=0.32)
    ax = pl.subplot(ax[:, :])
    
    pl.plot(dom, cuda[0])
    pl.plot(dom, broad0[0])
    ax.set_ylabel("approximated pdf")
    ax.set_xlabel("enrichment score")
    
    ax=gridspec.GridSpec(1, 1)
    ax.update(left=0.40, right=0.65)
    ax = pl.subplot(ax[:, :])
    
    pl.plot(dom, Cuda)
    pl.plot(dom, Broad0)
    ax.set_title(name, y=1.04)
    ax.set_ylabel("approximated cdf")
    ax.set_xlabel("enrichment score")
    
    ax=gridspec.GridSpec(1, 1)
    ax.update(left=0.74, right=0.95)
    ax = pl.subplot(ax[:, :])
    
    pl.plot(dom, Cuda-Broad0)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_ylabel("difference of approximated cdfs")
    ax.set_xlabel("enrichment score")
    
    bc.append(max(abs(Cuda-Broad0)))
    bb.append(max(abs(Broad0-Broad1)))
    
    pl.savefig("images/"+name+".pdf", dpi=300)
    pl.savefig("images/"+name+".png", dpi=300)
    pl.show()
    
print sum(map(lambda (x,y) : 1 if x < y else 0, zip(bc, bb)))
print min(bc), np.median(bc), max(bc), np.mean(bc), np.std(bc)
print min(bb), np.median(bb), max(bb), np.mean(bb), np.std(bb)

