import os
for i in range(1, 32):
	file_name = str(i) + "_combine.seq"
	input_address = "../train_and_test/" + file_name
	output_address = "../train_and_test/top100/" + file_name[:-4] + ".top100"
        print('nohup ./RNAsubopt --deltaEnergy 10000 --stochBT_en 100 < ' + input_address + ' > ' + output_address + " &")
