#!/usr/bin/env bash
echo "Start test"
echo "template base"
#template base
python3 crf.py base res_file.txt evalu.txt model.txt 2
python3 crf.py base res_file3.txt evalu3.txt model3.txt 3
python3 crf.py base res_file4.txt evalu4.txt model4.txt 4
python3 crf.py base res_file5.txt evalu5.txt model5.txt 5
python3 crf.py base res_file6.txt evalu6.txt model6.txt 6

#template unigram
echo "template unigram"
python3 crf.py unigram res_uni.txt eval_uni.txt model_uni.txt 2
python3 crf.py unigram res_uni3.txt eval_uni3.txt model_uni3.txt 3
python3 crf.py unigram res_uni4.txt eval_uni4.txt model_uni4.txt 4
python3 crf.py unigram res_uni5.txt eval_uni5.txt model_uni5.txt 5
python3 crf.py unigram res_uni6.txt eval_uni6.txt model_uni6.txt 6

#template window bigram
echo "template window bigram"
python3 crf.py window_bigram res_wb.txt eval_wb.txt model_wb.txt 2
python3 crf.py window_bigram res_wb3.txt eval_wb3.txt model_wb3.txt 3
python3 crf.py window_bigram res_wb4.txt eval_wb4.txt model_wb4.txt 4
python3 crf.py window_bigram res_wb5.txt eval_wb5.txt model_wb5.txt 5
python3 crf.py window_bigram res_wb6.txt eval_wb6.txt model_wb6.txt 6

#template window bigram 2
echo "template window bigram 2"
python3 crf.py window_bigram2 res_wb_2.txt eval_wb_2.txt model_wb_2.txt 2
python3 crf.py window_bigram2 res_wb_2-3.txt eval_wb_2-3.txt model_wb_2-3.txt 3
python3 crf.py window_bigram2 res_wb_2-4.txt eval_wb_2-4.txt model_wb_2-4.txt 4
python3 crf.py window_bigram2 res_wb_2-5.-txt eval_wb_2-5.txt model_wb_2-5.txt 5
python3 crf.py window_bigram2 res_wb_2-6.txt eval_wb_2-6.txt model_wb_2-6.txt 6

#template window
echo "template window"
python3 crf.py windows res_win.txt eval_win.txt model_win.txt 2
python3 crf.py windows res_win3.txt eval_win3.txt model_win3.txt 3
python3 crf.py windows res_win4.txt eval_win4.txt model_win4.txt 4
python3 crf.py windows res_win5.txt eval_win5.txt model_win5.txt 5
python3 crf.py windows res_win6.txt eval_win6.txt model_win6.txt 6

#template window bigram lemma pos 
echo "template window bigram lemma pos"
python3 crf.py windows_bi_lemma_pos res_wblp.txt eval_wblp.txt model_wblp.txt 2
python3 crf.py windows_bi_lemma_pos res_wblp3.txt eval_wblp3.txt model_wblp.txt 3
python3 crf.py windows_bi_lemma_pos res_wblp4.txt eval_wblp4.txt model_wblp4.txt 4
python3 crf.py windows_bi_lemma_pos res_wblp5.txt eval_wblp5.txt model_wblp5.txt 5
python3 crf.py windows_bi_lemma_pos res_wblp6.txt eval_wblp6.txt model_wblp6.txt 6




