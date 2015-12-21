import numpy as np
import array as ar

def clean(text):

    upper=text.find("///")
    return text[:upper].strip() if upper > 0 else text

def read_chip(chipfile):
    
    count, mapping = 0, {}

    with open(chipfile, "r") as f:
        for line in f:
            if "_at" in line:
                tokens = line.split("\t")
                spot, symbol = tokens[0], clean(tokens[10])
                if symbol == "":
                    print "no symbol for spot", spot
                else:
                    mapping[spot] = symbol
                    count += 1

    assert(count == len(mapping))
        
    return mapping

def map_array(arrayfile, mapping):

    count, array, titles = 0, {}, None

    with open(arrayfile, "r") as f:
        for line in f:
            if "_at" in line:
                tokens = line.replace('\n', '').replace('"', '').split('\t')
                spot = tokens[0]
                if spot in mapping:
                    key  = mapping[spot]
                    data = np.array(map(float, tokens[1:]))
                    if key in array:
                        data = np.maximum(data, array[key])
                        print "max pooled key", key
                    array[key] = data
                else:
                    print "cannot map spot", spot
            
            if "!Sample_title" in line:
                titles = line.replace('\n', '').replace('"', '').split("\t")[1:]

    return array, titles

def write_gct(filename, array, titles):

    with open(filename, "w") as f, open(filename+".bin", "wb") as g:
        f.write("#1.2\n")
        f.write("%s\t%s\n" % (len(array), len(titles)))
        f.write("NAME\tDESCRIPION\t%s\n" % "\t".join(titles))
        g.write(ar.array("L", [len(array), len(titles)]))

        for key in array:
            f.write("%s\tna\t%s\n" % (key.strip(), "\t".join(map(str,array[key]))))
            g.write(ar.array("f", array[key]))
            print "SYMBOLKEY", key
            

mapping = read_chip("raw/GPL570-13270")
array, titles = map_array("raw/GSE19429_series_matrix.txt", mapping)
write_gct("GSE19429_series.gct", array, titles)

print len(titles)



