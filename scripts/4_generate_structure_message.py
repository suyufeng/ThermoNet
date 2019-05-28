import sys
import os
import math
import numpy as np
import scipy.sparse
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', action='store', default='summary.tsv', help='log dir')
parser.add_argument('--output_dir', action='store', default='summary.tsv', help='log dir')
parser.add_argument('--num', action='store', default='summary.tsv', help='log dir')
parser.add_argument('--flag', action='store', default='summary.tsv', help='log dir')

father_path=os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")

args = parser.parse_args()
fin = open(args.input_dir, "r")
flag = False
if args.flag == 'True':
    flag = True
# fout = open(father_path + "/data/matrix.txt", "w")


dic = {}

num_seq = 0

left = []


max_length = 101
sample_num = int(args.num)


def ring(left, right, message_array, edge, sign):
    if edge[left + 1] == right - 1:
        ring(left + 1, right - 1, message_array, edge, sign)
        return
    next_start = left + 1
    ring_list = []
    ring_list.append(left)
    num = 0
    while next_start <= right - 1:
        ring_list.append(next_start)
        if edge[next_start] == 0:
            next_start = next_start + 1
        else:
            ring(next_start, edge[next_start], message_array, edge, sign)
            ring_list.append(edge[next_start])
            next_start = edge[next_start] + 1
            num = num + 1
    if num == 1:
        if edge[right - 1] != 0 or edge[left + 1] != 0:
            return
    if edge[right] == 0:
        ring_list.append(right)
    num = min(num, 2)
    for i in ring_list:
        if edge[i] != 0:
            continue
        message_array[num][i - 1] = 1
        sign[i - 1] = 1
    return


def get_feature(list, length, index):
    edge = np.zeros([max_length + 1], dtype=np.int16)
    for i in list:
        x = i[0]
        y = i[1]
        edge[x + 1] = y + 1
        edge[y + 1] = x + 1
    message_array = np.zeros([9, max_length], dtype=np.float32)
    # return message_array[:, st[index, 0]:st[index, 0] + 100]
    sign = np.zeros([max_length + 1], dtype=np.int16)
    start_id = 0
    end_id = 0

    for i in range(1, max_length + 1):
        if edge[i] != 0:
            start_id = i
            end_id = edge[i]
            break
    ring(start_id, end_id, message_array, edge, sign)
    for i in range(0, length):
        if edge[i + 1] != 0:
            message_array[3][i] = 1
        else:
            if sign[i] == 0:
                message_array[4][i] = 1
    for i in range(length, max_length):
        message_array[4][i] = 1
    for j in range(4):
        message_array[5 + j][0] = message_array[5 + j][max_length - 1] = 0
        for i in range(1, max_length - 1):
            if message_array[j][i - 1] + message_array[j][i + 1] == 1 and message_array[j][i] == 1:
                message_array[5 + j][i] = 1
            else:
                message_array[5 + j][i] = 0

        for i in range(1, max_length - 1):
            if message_array[j + 5][i - 1] + message_array[j + 5][i + 1] >= 1 and message_array[j][i] == 1 and message_array[5 + j][i] == 0:
                message_array[5 + j][i] += 0.5
    # print(message_array[:, st[index, 0]:st[index, 0] + 100].shape)
    return message_array[:, :]

def change_matrix(list):
    a = np.zeros([max_length , max_length])
    for i in list:
        x = i[0]
        y = i[1]
        a[x][y] = 1. / 100
        a[y][x] = 1. / 100

def bracket_match(pattern):
    length = len(pattern)
    list_edge = []
    for i in range(length):
        if pattern[i] == '.':
            continue
        if pattern[i] == '(':
            left.append(i)
            continue
        left_one = left[len(left) - 1]
        list_edge.append([left_one, i])
        left.pop()
    return list_edge

i = -1

for line in fin:
    a = line[:-1]
    if line[0] == '>' or (line[0] >= 'A' and line[0] <= 'Z'):
        continue
    i += 1
    if i % sample_num == 0:
        dic[i // sample_num] = []
        a = a.split(' ')[0]
        dic[i // sample_num].append(a)
    	continue
#    if i > sample_num:
#	break
    a = a.split(' ')[0]
    dic[i // sample_num].append(a)
    num_seq = i // sample_num
list = []
for i in range(num_seq + 1):
    tmp_list = []
#    print(i)
    for index, j in enumerate(dic[i]):
#	print(j)
        edge_list = bracket_match(j)
#	continue
        if flag == True:
            tmp_list.append(get_feature(edge_list, len(j), i))
        else:
            list.append(get_feature(edge_list, len(j), i))
#    continue
    if flag == True:
        tmp_list = np.array(tmp_list)
        tmp_list = np.reshape(tmp_list, [-1, sample_num, 9 * max_length])
        tmp_list = np.mean(tmp_list, axis=1).astype(np.float32)
        list.append(tmp_list)

result = np.array(list)
result.astype(np.float32)

if flag == True:
    result = np.reshape(result, [-1, 9 * max_length])
    np.save(args.output_dir, result)
else:
    result = np.reshape(result, [-1, sample_num * 9 * max_length])
    t = scipy.sparse.csr_matrix(result)
    scipy.sparse.save_npz(args.output_dir, t)
