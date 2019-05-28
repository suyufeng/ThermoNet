import os

t = "invivo_1"

for i in range(2, 32):
    t += " invivo_" + str(i)

list = t.split(' ')

k = "../log/result_sample"

for i in list:
    if (os.path.isfile(k + "/" + i + "_CNN_yaron.yml")):
        f = open(k + "/" + i + "_CNN_yaron.yml", "r")
        line = f.readline()
        line = f.readline()[:-2]
        t = line.split(", ")
        for i in range(len(t)):
            t[i] = t[i].split(": ")[1]
        print(str(t[0]))
    else:
        print(" ")
