import os
import random

# files 
FILE_TRAIN = "dataset/NLSPARQL.train.data"
FILE_TEST = "dataset/NLSPARQL.test.data"
TRAIN_FEATS = "dataset/NLSPARQL.train.feats.txt"
TEST_FEATS = "dataset/NLSPARQL.test.feats.txt"

#variables
#base list
word_list = []
iob_list = []
line_list = []
sentence_list = []
# lemma, pos list
lp_list = []
lp_line_list = []
lp_sentence_list = []
# word, lemma, pos list
wlp_list = []
wlp_line_list = []
wlp_sentence_list = []
# word, lemma list
wl_list = []
wl_line_list = []
wl_sentence_list = []
# word, pos list
wp_list = []
wp_line_list = []
wp_sentence_list = []

#create a file for training
with open(FILE_TRAIN) as file1, open(TRAIN_FEATS) as file2:
    for x, y in zip(file1, file2):
        # w1 in form word , IOB 	
        w1 = x.strip().split()
        # w2 in form word , POS , lemma 
        w2 = y.strip().split()
        #get the value
        print_str = ""
        # prefix and suffix extraction and add the list
        if len(w1)>0 and len(w2)>0:
            word_list.append(w1[0])
            iob_list.append(w1[1])
            line_list.append((w1[0],w1[1]))
            lp_list.append((w2[2], w2[1]))
            lp_line_list.append((w2[2], w2[1], w1[1]))
            wlp_list.append((w1[0],w2[2],w2[1]))
            wlp_line_list.append((w1[0],w2[2],w2[1],w1[1]))
            wl_list.append((w1[0], w2[2]))
            wl_line_list.append((w1[0], w2[2], w1[1]))
            wp_list.append((w1[0], w2[1]))
            wp_line_list.append((w1[0], w2[1],w1[1]))
        else:
            if len(line_list) > 0:
                sentence_list.append(line_list)
                line_list = []
            if len(lp_line_list) > 0:
                lp_sentence_list.append(lp_line_list)
                lp_line_list = []
            if len(wlp_line_list) > 0:
                wlp_sentence_list.append(wlp_line_list)
                wlp_line_list = []
            if len(wl_line_list) > 0:
                wl_sentence_list.append(wl_line_list)
                wl_line_list = []
            if len(wp_line_list) > 0:
                wp_sentence_list.append(wp_line_list)
                wp_line_list = []

#Create a Lexicon
#delete the duplicate into the list
unique_w_list = list(set(word_list))
unique_iob_list = list(set(iob_list))
unique_lp_list = list(set(lp_list))
unique_wlp_list = list(set(wlp_list))
unique_wl_list = list(set(wl_list))
unique_wp_list = list(set(wp_list))

#create file to unique lexicon
with open("lexicon/unique_w_lexicon", "w") as wlexicon:
    counter = 0
    for element in unique_w_list:
        wlexicon.write(element + " " + str(counter) + "\n")
        counter += 1
    wlexicon.write("<UNK> " + str(counter) + "\n" )

with open("lexicon/unique_iob_lexicon", "w") as ioblexicon:     
    #create file unique iob lexicon
    counter = 0
    for element in unique_iob_list:
        ioblexicon.write(element + " " + str(counter) + "\n")
        counter += 1
    ioblexicon.write("<UNK> " + str(counter) + "\n" )

with open("lexicon/unique_lp_lexicon", "w") as lplexicon:
    #create unique lemma, pos lexicon
    counter = 0
    for element in unique_lp_list:
        # separated by '_' so that can eventually be split
        lplexicon.write(element[0] + "_" + element[1] + " " + str(counter) + "\n")
        counter += 1
    lplexicon.write("<UNK> " + str(counter) + "\n")

with open("lexicon/unique_wlp_lexicon", "w") as wlplexicon:
    #create unique word, lemma, pos list
    counter = 0
    for element in unique_wlp_list:
        # separated by '_' so that can eventually be split
        wlplexicon.write(element[0] + "_" + element[1] + "_" + element[2] + " " + str(counter) + "\n")
        counter += 1
    wlplexicon.write("<UNK> " + str(counter) + "\n")

with open("lexicon/unique_wl_lexicon", "w") as wllexicon:
    #create unique word lemma
    counter = 0
    for element in unique_wl_list:
        # separated by '_' so that can eventually be split
        wllexicon.write(element[0] + "_" + element[1] + " " + str(counter) + "\n")
        counter += 1
    wllexicon.write("<UNK> " + str(counter) + "\n")

with open("lexicon/unique_wp_lexicon", "w") as wplexicon:
    #create unique word pos list
    counter = 0
    for element in unique_wp_list:
        # separated by '_' so that can eventually be split
        wplexicon.write(element[0] + "_" + element[1] + " " + str(counter) + "\n")
        counter += 1
    wplexicon.write("<UNK> " + str(counter) + "\n")



# create file of test
test_lp_line = []
test_lp_set = []
test_wlp_line = []
test_wlp_set = []
test_wl_line = []
test_wl_set = []
test_wp_line = []
test_wp_set = []

with open("dataset/NLSPARQL.test.data", "r") as file1, open("dataset/NLSPARQL.test.feats.txt") as file2:
    for x, y in zip(file1, file2):
        # data_tokens in form < word , IOB >
        w1 = x.strip().split()
        # feature_tokens in form < word , POS , lemma >
        w2 = y.strip().split()
        # prefix and suffix extraction (based on user's parameter)
        if len(w1) > 0 and len(w2) > 0:
            iob_list.append(w1[1])
            test_lp_line.append((w2[2],w2[1],w1[1]))
            test_wlp_line.append((w1[0],w2[2],w2[1],w1[1]))
            test_wl_line.append((w1[0],w2[2],w1[1]))
            test_wp_line.append((w1[0], w2[1], w1[1]))
        else:
            if len(test_lp_line) > 0:
                test_lp_set.append(test_lp_line)
                test_lp_line = []
            if len(test_wlp_line) > 0:
                test_wlp_set.append(test_wlp_line)
                test_wlp_line = []
            if len(test_wl_line) > 0:
                test_wl_set.append(test_wl_line)
                test_wl_line = []
            if len(test_wp_line) > 0:
                test_wp_set.append(test_wp_line)
                test_wp_line = []


# implement the validation
#
random.shuffle(sentence_list)
random.shuffle(lp_sentence_list)
random.shuffle(wl_sentence_list)
random.shuffle(wp_sentence_list)

#create the training set and the validation set to alla sentences
training_set = sentence_list[:int(len(sentence_list) * 0.30)]
validation_set = sentence_list[int(len(sentence_list) * 0.30):]

training_lp_set = lp_sentence_list[:int(len(lp_sentence_list)*0.30)]
validation_lp_set = lp_sentence_list[int(len(lp_sentence_list)*0.30):]

training_wlp_set = wlp_sentence_list[:int(len(wlp_sentence_list)*0.30)]
validation_wlp_set = wlp_sentence_list[int(len(wlp_sentence_list)*0.30):]

training_wl_set = wl_sentence_list[:int(len(wl_sentence_list)*0.30)]
validation_wl_set = wl_sentence_list[int(len(wl_sentence_list)*0.30):]

training_wp_set = wp_sentence_list[:int(len(wp_sentence_list)*0.30)]
validation_wp_set = wp_sentence_list[int(len(wp_sentence_list)*0.30):]

# save the training and the validation set in a file
with open("training/training_set", "w") as training_set_file:
    for list in training_set:
        for element in list:
            training_set_file.write(str(element[0]) + " " + str(element[1]) + "\n")
        training_set_file.write("\n")
with open("validation/validation_set", "w") as validation_set_file:
    for list in validation_set:
        for element in list:
            validation_set_file.write(str(element[0]) + " " + str(element[1]) + "\n")
        validation_set_file.write("\n")

# save the training, the test and the validation set in a file (<lemma,pos>)
with open("training/training_lp_set", "w") as training_lp_set_file:
    for list in training_lp_set:
        for element in list:
            training_lp_set_file.write(element[0] + "_" + element[1] + " " + element[2] + "\n")
        training_lp_set_file.write("\n")
with open("validation/validation_lp_set", "w") as validation_lp_set_file:
    for list in validation_lp_set:
        for element in list:
            validation_lp_set_file.write(element[0] + "_" + element[1] + " " + element[2] + "\n")
        validation_lp_set_file.write("\n")
with open("test/test_lp_set", "w") as test_lp_set_file:
    for list in test_lp_set:
        for element in list:
            test_lp_set_file.write(element[0] + "_" + element[1] + " " + element[2] + "\n")
        test_lp_set_file.write("\n")

# save the training, the test and the validation set in a file (<word,lemma,pos>)
with open("training/training_wlp_set", "w") as training_wlp_set_file:
    for list in training_wlp_set:
        for element in list:
            training_wlp_set_file.write(element[0] + "_" + element[1] + "_" + element[2] + " " + element[3] + "\n")
        training_wlp_set_file.write("\n")
with open("validation/validation_wlp_set", "w") as validation_wlp_set_file:
    for list in validation_wlp_set:
        for element in list:
            validation_wlp_set_file.write(element[0] + "_" + element[1] + "_" + element[2] + " " + element[3] + "\n")
        validation_wlp_set_file.write("\n")
with open("test/test_wlp_set", "w") as test_wlp_set_file:
    for list in test_wlp_set:
        for element in list:
            test_wlp_set_file.write(element[0] + "_" + element[1] + "_" + element[2] + " " + element[3] + "\n")
        test_wlp_set_file.write("\n")

# save the training, the test and the validation set in a file (<word,lemma>)
with open("training/training_wl_set", "w") as training_wl_set_file:
    for list in training_wl_set:
        for element in list:
            training_wl_set_file.write(element[0] + "_" + element[1] + " " + element[2] + "\n")
        training_wl_set_file.write("\n")
with open("validation/validation_wl_set", "w") as validation_wl_set_file:
    for list in validation_wl_set:
        for element in list:
            validation_wl_set_file.write(element[0] + "_" + element[1] + " " + element[2] + "\n")
        validation_wl_set_file.write("\n")
with open("test/test_wl_set", "w") as test_wl_set_file:
    for list in test_wl_set:
        for element in list:
            test_wl_set_file.write(element[0] + "_" + element[1] + " " + element[2] + "\n")
        test_wl_set_file.write("\n")

# save the training, the test and the validation set in a file (<word,pos>)
with open("training/training_wp_set", "w") as training_wp_set_file:
    for list in training_wp_set:
        for element in list:
            training_wp_set_file.write(element[0] + "_" + element[1] + " " + element[2] + "\n")
        training_wp_set_file.write("\n")
with open("validation/validation_wp_set", "w") as validation_wp_set_file:
    for list in validation_wp_set:
        for element in list:
            validation_wp_set_file.write(element[0] + "_" + element[1] + " " + element[2] + "\n")
        validation_wp_set_file.write("\n")
with open("test/test_wp_set", "w") as test_wp_set_file:
    for list in test_wp_set:
        for element in list:
            test_wp_set_file.write(element[0] + "_" + element[1] + " " + element[2] + "\n")
        test_wp_set_file.write("\n")


