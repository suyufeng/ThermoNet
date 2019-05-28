import os
for i in range(1, 32):
	file_name = str(i) + "_combine.seq"
	input_address = "../train_and_test/top100/" + file_name[:-4] + ".top100"
	output_address = "../train_and_test/top5/" + file_name[:-4] + ".top5"
        print('nohup python 5_generate_top5.py --input_dir ' + input_address + ' --output_dir ' + output_address + " &")
