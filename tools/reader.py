import numpy as np
import pylab as pl
import array as ar
import sys

filename = sys.argv[1]
tokens = filename.split(".")[-2].split("_")

print "Opening file:", filename
with open(filename, "rb") as f:
    if tokens[-1] == "64":
        data = np.array(ar.array("d", f.read()))
    elif tokens[-1] == "32":
        data = np.array(ar.array("f", f.read()))
    else:
        print "ERROR: Incompatible file name, expecting 32 or 64 at the end"
        sys.exit(1)

try:
    num_perms = int(tokens[-2])
    num_paths = int(tokens[-3])
except:
    print "ERROR: Cannot parse dimension of the matrix"
    sys.exit(2)

data=data.reshape((num_paths, num_perms))

# plot all scores in a table
pl.title("enrichment scores")
pl.imshow(data, aspect="auto", interpolation="nearest")
pl.xlabel("permutations")
pl.ylabel("gene set id")
pl.colorbar()
pl.show()

# visualize histograms
for path in range(num_paths):
    H = pl.hist(data[path], bins=int(np.sqrt(num_perms)), normed=False)
    pl.plot([+data[path,0], +data[path,0]], [0, max(H[0])], c="r", linewidth=3)
    pl.plot([-data[path,0], -data[path,0]], [0, max(H[0])], c="r", linewidth=3)
    pl.title("Gene Set %s with Enrichment Score %s" % (path, data[path,0]))
    pl.xlabel("enrichment scores over %s permutations" % num_perms)
    pl.ylabel("number of occurrences")
    pl.show()
