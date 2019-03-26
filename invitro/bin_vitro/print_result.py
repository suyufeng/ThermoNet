import os
t = "RNCMPT00004 RNCMPT00016 RNCMPT00017 RNCMPT00019 RNCMPT00023 RNCMPT00025"

list = t.split(' ')

k = "results_kmer_3_f5"

for i in list:
    if (os.path.isfile(k + "/" + i + "_CNN_attention.yml")):
        f = open(k + "/" + i + "_CNN_attention.yml", "r")
        line = f.readline()
        line = f.readline()[:-2]
        t = line.split(", ")
        for i in range(len(t)):
            t[i] = t[i].split(": ")[1]
        print(str(t[0]) + "\t" + str(t[1]))
    else:
        print(" ")