import sys
import os 
from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
from os.path import isfile, join
import string
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from os import listdir
from time import time
import re
import numpy as np
import pickle
import nltk
import csv
from helper import *

if len(sys.argv)<2:
	print('Not enough arguments, should be ','python '+sys.argv[0]+' filename')
	print('Check usage in readme.txt')
	exit()

if '20_newsgroups' in sys.argv[1]:
	my_path = '../data/20_newsgroups_Known/'   		#path to data folder
	my_path_novel= '../data/20_newsgroups_novel/'  #path to data folder
	outfile='../data/dt_newsgroup_cuda.csv'
if 'BBC' in sys.argv[1]:
	my_path = '../data/BBC_Known/'   		#path to data folder
	my_path_novel= '../data/BBC_novel/'  #path to data folder
	outfile='../data/dt_bbc_cuda.csv'

def loadfolder(path_dataset):
    #creating a list of folder names to make valid pathnames later
    folders = [f for f in listdir(path_dataset)]
    #creating a 2D list to store list of all files in different folders
    files = []
    for folder_name in folders:
        folder_path = join(path_dataset, folder_name)
        files.append([f for f in listdir(folder_path)])
    pathname_list = []
    for fo in range(len(folders)):
        for fi in files[fo]:
            pathname_list.append(join(path_dataset, join(folders[fo], fi)))
    Y = []
    for folder_name in folders:
        folder_path = join(path_dataset, folder_name)
        num_of_files= len(listdir(folder_path))
        for i in range(num_of_files):
            Y.append(folder_name)
    return pathname_list, Y

#making an array containing the classes each of the documents belong to
pathname_list, Y= loadfolder(my_path)
pathname_list_novel, Y_novel= loadfolder(my_path_novel)

label_encoder = preprocessing.LabelEncoder()
Y_novel = label_encoder.fit_transform(Y_novel)

label_encoder = preprocessing.LabelEncoder()
Y = label_encoder.fit_transform(Y)

## Split
doc_train, doc_test, Y_train, Y_test = train_test_split(pathname_list, Y, random_state=42,shuffle=True, test_size=0.50)

for document in doc_train:
    list_of_words.append(flatten(tokenize(document)))
for document_test in doc_test:
    list_of_words_test.append(flatten(tokenize(document_test)))
for document_novel in pathname_list_novel:
    list_of_words_novel.append(flatten(tokenize(document_novel)))


np_list_of_words= np.asarray(list_of_words)
np_list_of_words_test= np.asarray(list_of_words_test)
np_list_of_words_word_idx= np.asarray(flatten(list_of_words))
np_list_of_words_novel=np.asarray(list_of_words_novel)

words, counts = np.unique(np_list_of_words_word_idx, return_counts=True)
freq, wrds = (list(i) for i in zip(*(sorted(zip(counts, words), reverse=True))))

#create dict for vocab
if not(os.path.exists('./vocab.pkl')):
  word_set=set(wrds)
  word_idx = dict((c, i + 1) for i, c in enumerate(word_set))
  np_word_idx= np.asarray(word_idx)
  output= open("vocab.pkl", "wb")
  pickle.dump(word_idx, output)
  output.close()
  
saved= open("vocab.pkl", "rb")
word_idx_saved= pickle.load(saved)
saved.close()
print("word_idx",len(word_idx_saved))

#Encode into binary features
def encoding_sent(text):
    feature_set = np.zeros((len(text), len(word_idx_saved)), dtype=np.uint32)
    tnum=0
    for t in text:
        for w in t:
            if (w in word_idx_saved):
                idx = word_idx_saved[w]
                feature_set[tnum][idx-1] = 1
        tnum += 1
    return feature_set

X_train = encoding_sent(np_list_of_words)
X_test = encoding_sent(np_list_of_words_test)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)
X_novel= encoding_sent(np_list_of_words_novel)
Y_novel=np.asarray(Y_novel)


## Tsetlin machine parameteres
if '20_newsgroups' in sys.argv[1]:
	NUM_CLAUSES = 10000  
	T = 400	  
	s = 25.0     
if 'BBC' in sys.argv[1]:
	NUM_CLAUSES = 5000  
	T = 100	  
	s = 15.0    
       

#passing parameters to tsetlin machine
tm = MultiClassTsetlinMachine(NUM_CLAUSES, T, s, append_negated=False)

#start training and testing
print("\nAccuracy over 100 epochs:\n")
for epo in range(100):
  start = time()
  tm.fit(X_train, Y_train, epochs=1, incremental=True) 
  stop = time()
  result = 100*(tm.predict(X_train) == Y_train).mean()
  print("#%d Training Accuracy: %.2f%% (%.2fs)" % (epo+1, result, stop-start))
  start = time()
  result_test = 100*(tm.predict(X_test) == Y_test).mean()
  stop = time()
  score_novel= tm.score(X_novel)
  score_known= tm.score(X_test)

  print("#%d Testing Accuracy: %.2f%% (%.2fs)" % (epo+1, result_test, stop-start)) 

A_novel= score_novel[0]
B_novel=score_novel[1]
label= np.zeros(len(A_novel), dtype=np.int32)
with open(outfile, 'w',newline='') as file:
    writer=csv.writer(file)
    writer.writerows(zip(A_novel, B_novel, label))

A_known= score_known[0]
B_known= score_known[1]
label= np.ones(len(A_known), dtype=np.int32)
with open(outfile, 'a+',newline='') as file:
    writer=csv.writer(file)
    writer.writerows(zip(A_known, B_known, label))