import os
st = 28
ed = st + 1
for i in range(st, ed):
	file_name = str(i) + "_combine"
	input_address = "../train_and_test/top5/" + file_name + ".top5"
	output_address = "../train_and_test/top5/" + file_name + ".map.top5"
        #os.system("./RNAsubopt --deltaEnergy 10000 --stochBT_en 100 < ' + input_address + ' > ' + output_address")
	print("python 4_generate_structure_message.py" + " --input_dir " + input_address + " --output_dir " + output_address + " --num 5 --flag False")
