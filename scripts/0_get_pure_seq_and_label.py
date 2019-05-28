for i in range(1, 32):
	f = open("../train_and_test/" + str(i) + "_combine.seq", "r")
	flabel = open("../train_and_test/" + str(i) + ".label", "w")
	fseq = open("../train_and_test/" + str(i) + ".pure.seq", "w")
	lable = 0
	for index, line in enumerate(f):
		if index % 3 == 0:
			label = line.split("class:")[1]
		elif index % 3 == 1:
			flabel.write(label)
			fseq.write(line[:-1])
		else:
			fseq.write(line)
