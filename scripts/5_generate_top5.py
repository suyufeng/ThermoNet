import numpy as np
import os
import argparse


# python calc_train.py --input_dir ../data/results/PARCLIP_PUM2.fullseq.fa.sub --output_dir ../data/results/PARCLIP_PUM2.fullseq.fa.sub.top5


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', action='store', default='summary.tsv', help='log dir')
parser.add_argument('--output_dir', action='store', default='summary.tsv', help='log dir')

father_path=os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")

args = parser.parse_args()
fin = open(args.input_dir, "r")
t = {}
bar = 5

res_list = []

fout = open(args.output_dir, "w")

pred_list = []

index = -1
for index_1, line in enumerate(fin):

    if line[0] == '>':
        continue
    index += 1

    if index % 101 == 0 and index != 0:
        num_list = []
        list = []
        for (k, v) in t.items():
            num_list.append(v)
        num_list.sort(reverse=True)
        res = 0
        total_num = 0
        sum = 0
        base_line = 0
        if len(num_list) > bar:
            base_line = num_list[bar]
        for i in range(min(bar, len(num_list))):
            sum += num_list[i]
            if num_list[i] > base_line:
                total_num += 1
        res_list.append(total_num)
        last_seq = " "
        for (k, v) in t.items():
            if v > base_line or (v == base_line and total_num < 5):
                fout.write(k + "\n")
                last_seq = k
                list.append(1. * v / 100)
                if len(num_list) > bar and v == num_list[bar]:
                    total_num += 1

        pred = 1. * list[len(list) - 1] / (5 - total_num + 1)
        list[len(list) - 1] = pred
        for i in range(total_num, 5):
            fout.write(last_seq + "\n")
            list.append(pred)

        list.append(1. * (100 - sum) / 100)
        pred_list.append(np.array(list))
        t.clear()

    if index % 101 == 0:
        continue

    a = line.split(" ")
    if a[0] in t:
        t[a[0]] = t[a[0]] + 1
    else:
        t[a[0]] = 1

num_list = []
list = []
for (k, v) in t.items():
    num_list.append(v)
num_list.sort(reverse=True)
res = 0
total_num = 0
sum = 0
base_line = 0
if len(num_list) > bar:
    base_line = num_list[bar]
for i in range(min(bar, len(num_list))):
    sum += num_list[i]
    if num_list[i] > base_line:
        total_num += 1
res_list.append(total_num)
last_seq = " "
for (k, v) in t.items():
    if v > base_line or (v == base_line and total_num < 5):
        fout.write(k + "\n")
        last_seq = k
        list.append(1. * v / 100)
        if len(num_list) > bar and v == num_list[bar]:
            total_num += 1

pred = 1. * list[len(list) - 1] / (5 - total_num + 1)
list[len(list) - 1] = pred
for i in range(total_num, 5):
    fout.write(last_seq + "\n")
    list.append(pred)
list.append(1. * (100 - sum) / 100)

pred_list.append(np.array(list))
t.clear()


pred_list = np.array(pred_list)
np.save(args.output_dir + '.prob', pred_list)
