for i in range(1, 32):
	fin = open("../train_and_test/" + str(i) + "_training.seq", "r")
	fout = open("../train_and_test/" + str(i) + "_combine.seq", "w")
	for j in range(2):
		for line in fin:
			fout.write(line)
		fin = open("../train_and_test/" + str(i) + "_test.seq", "r")

