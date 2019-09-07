# Lus-Project2
Language Understanding System - Project 2

The goal is to develop for the movie domain a Spoken Language Understanding (SLU) Module with discriminative model and develop the recurrent neural network using NL-SPARQL Dataset.

The project contains two folder, CRF and RNN.

## CFR folder contains:

+ Dataset
+ File crf.py This file is the start program. To generate the results, elaborate the training and test sets and to create the other necessary files.
+ Template: in this folder found the templates used.


To run this part of project write in the terminal: python3 crf.py unigram res_uni4.txt eval_uni4.txt model_uni4.txt 4


## RNN folder contain:

### Elman
+ Dataset
+ File RNN.py This file is the start program. his script will generate all the necessary files in the folder data.
+ rnn_models: folder where found the different configuration used
+ File result.py. File that generate the result.

To run this part of project write in the terminal:
python rnn_slu/lus/rnn_elman_train.py training/training_wp_set validation/validation_wp_set lexicon/unique_wp_lexicon lexicon/unique_iob_lexicon rnn_models/config4.cfg results/config4_wp_elman  

python rnn_slu/lus/rnn_elman_test.py config4_wp_elman test/test_wp_set lexicon/unique_wp_lexicon lexicon/unique_iob_lexicon rnn_models/config4.cfg out_config4_wp_elman 

python result.py


### Jordan
+ Dataset
+ File RNN.py This file is the start program. his script will generate all the necessary files in the folder data.
+ rnn_models: folder where found the different configuration used
+ File result.py. File that generate the result.

To run this part of project write in the terminal:
python rnn_slu/lus/rnn_jordan_train.py training/training_set validation/validation_set lexicon/unique_w_lexicon lexicon/unique_iob_lexicon rnn_models/config4.cfg results/config4_jordan 

python rnn_slu/lus/rnn_jordan_test.py config4_wp_jordan test/test_wp_set lexicon/unique_wp_lexicon lexicon/unique_iob_lexicon rnn_models/config4.cfg out_config4_wp_jordan 

python result.py

### RNN_SLU
In this folder a set of scripts to train and test Jordan RNNs and Elman RNNs can be found. All the scripts in this folder have been provided during the classes.
