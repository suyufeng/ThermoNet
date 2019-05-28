import numpy as np

dic = {}
dic['A'] = dic['N'] = 0
dic['G'] = 1
dic['C'] = 2
dic['T'] = dic['U'] = 3



for i in range(1, 32):
	f = open("../train_and_test/" + str(i) + ".pure.seq", "r")
	ans = {}
	for j in range(5):
		ans[j] = []

	for line in f:
		line = line[:-1]
		length = len(line)
		base = 0
		k_mer = np.zeros([5, length])
		for j in range(5):
			base *= 4
			for pos in range(length - j):
				hash_value = 0
				for _ in range(pos, pos + j + 1):
					hash_value *= 4
					hash_value += dic[line[_]]
				hash_value += base
				k_mer[j, pos] = hash_value
			for pos in range(length - j, length):
				k_mer[j, pos] = base
			if base == 0:
				base += 1	
			ans[j].append(k_mer[j, :])
	
	for j in range(5):
		fout = "../train_and_test/n_gram/" + str(i) + "." + str(j + 1) + "_mer"
		np.save(fout, np.array(ans[j]))
