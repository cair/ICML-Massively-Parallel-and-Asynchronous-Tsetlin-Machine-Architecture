import numpy as np
import keras
from keras.datasets import imdb
from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
from time import time 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

MAX_NGRAM = 2

NUM_WORDS=5000
INDEX_FROM=2 

FEATURES=5000

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

# Produce N-grams

id_to_word = {value:key for key,value in word_to_id.items()}

vocabulary = {}
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])
	
	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			
			if phrase in vocabulary:
				vocabulary[phrase] += 1
			else:
				vocabulary[phrase] = 1

# Assign a bit position to each N-gram (minimum frequency 10) 

phrase_bit_nr = {}
bit_nr_phrase = {}
bit_nr = 0
for phrase in vocabulary.keys():
	if vocabulary[phrase] < 10:
		continue

	phrase_bit_nr[phrase] = bit_nr
	bit_nr_phrase[bit_nr] = phrase
	bit_nr += 1

# Create bit representation

X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_train[i,phrase_bit_nr[phrase]] = 1

	Y_train[i] = train_y[i]

X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)

for i in range(test_y.shape[0]):
	terms = []
	for word_id in test_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_test[i,phrase_bit_nr[phrase]] = 1				

	Y_test[i] = test_y[i]

print("Selecting features...")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

## Tsetlin machine parameteres
NUM_CLAUSES=7000
T=80
s=27.0


#passing parameters to tsetlin machine
tm = MultiClassTsetlinMachine(NUM_CLAUSES, T, s)

#start training and testing
print("\nAccuracy over 100 epochs:\n")
for epo in range(100):
	start= time()
	tm.fit(X_train, np.asarray(Y_train), epochs=1, incremental=True)
	stop= time()
	result = 100*(tm.predict(X_train) == Y_train).mean()
	print("#%d Training Accuracy: %.2f%% (%.2fs)" % (epo+1, result, stop-start))
	start = time()
	pred=tm.predict(X_test)
	stop = time()
	result_test = 100*(pred == Y_test).mean()
	print("#%d Testing Accuracy: %.2f%% (%.2fs)" % (epo+1, result_test, stop-start))