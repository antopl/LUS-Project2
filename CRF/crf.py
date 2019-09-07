import string
import os
import sys

# files
FILE_TRAIN = "dataset/NLSPARQL.train.data"
FILE_TEST = "dataset/NLSPARQL.test.data"
TRAIN_FEATS = "dataset/NLSPARQL.train.feats.txt"
TEST_FEATS = "dataset/NLSPARQL.test.feats.txt"

#modality to start the project
if len(sys.argv) < 3:
    print('Please, insert the parameter: ')
    print('P1: template_crf, P2: name result file, P3: name evaluation file, P4 name model file, P5: int number prefix-suffix')
    exit(0)

#create a file that contain IOB and grammar tags for training
output = open("data_tmp/train_complete.data", "w")
with open(FILE_TRAIN) as file1, open(TRAIN_FEATS) as file2:
	for x, y in zip(file1, file2):
		# w1 in form word , IOB	
		w1 = x.strip().split()
		# w2 in form word , POS , lemma
		w2 = y.strip().split()
		#get the value
		print_str = ""
		if len(w1)>0 and len(w2)>0:
			if (len(w1[0]) > int(sys.argv[5])):
				pre = w1[0][0:int(sys.argv[5])]
				post = w1[0][-int(sys.argv[5])]
			else:
				pre = "NOPre"
				post = "NOPost"

			output.write(w1[0] + " " + w2[1] + " " + w2[2] + " "+ pre + " " + post +" "+ w1[1]+"\n")
		else:
			output.write("\n")

		

output.close()

#creating test files for CRF
output2 = open("data_tmp/test_complete.data", "w")
with open(FILE_TEST) as file1, open(TEST_FEATS) as file2:
	for x, y in zip(file1, file2):
		# w1 in form < word , IOB >	
		w1 = x.strip().split()
		# w2 in form < word , POS , lemma >
		w2 = y.strip().split()
		#get the value
		print_str = ""
		if len(w1)>0 and len(w2)>0:
			if (len(w1[0]) > int(sys.argv[5])):
				pre = w1[0][0:int(sys.argv[5])]
				post = w1[0][-int(sys.argv[5])]
			else:
				pre = "NOPre"
				post = "NOPost"
			output2.write(w1[0] + " " + w2[1] + " " + w2[2] + " "+ pre + " " + post +" " + w1[1]+"\n")
		else:
			output2.write("\n")
		

output2.close()

#call the CRF parameter with template, data and model
os.system('crf_learn '+ 'template/'+str(sys.argv[1]) + ' data_tmp/train_complete.data '+ 'model/'+str(sys.argv[4]))
os.system('crf_test -m ' + 'model/'+str(sys.argv[4]) + ' data_tmp/test_complete.data >'+ 'results/'+str(sys.argv[2]))	
os.system(' perl evaluation/conlleval.pl -d \'\s\' < ' + 'results/'+str(sys.argv[2]) + ' > ' + 'evaluation/'+str(sys.argv[3]) )

