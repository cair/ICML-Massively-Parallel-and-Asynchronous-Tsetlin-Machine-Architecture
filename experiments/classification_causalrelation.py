from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction import stop_words
import numpy as np
from nltk.util import ngrams,everygrams
import string
from time import time 

inp='../data/training_cause_effect.txt'

sents=[]
labels=[]
all_words=[]

stop=stop_words.ENGLISH_STOP_WORDS

def encode_sentences(txt):
	feature_set=np.zeros((len(txt), len(word_set)+1),dtype=int)
	tnum=0
	for t in txt:
		s_words=t[1:]+list(set(list(everygrams(t[1:], min_len=2,max_len=2))))
		for w in s_words:
			idx=word_idx[w]
			feature_set[tnum][idx]=1
		feature_set[tnum][-1]=t[0]
		tnum+=1
	return feature_set

maxlen=0
lcnt=0

for line in open(inp).readlines():
	if lcnt>0:
		line=line.replace('\n','').replace(',','').split('\t')
		line[0]=line[0].lower()
		line[0]=line[0].translate(str.maketrans('','',string.punctuation))
		words=line[0].split(' ')
		bl=list(set(list(everygrams(words, min_len=2,max_len=2))))
		all_words+=words+bl
		words.insert(0,lcnt)
		sents.append(words)
		labels.append(int(line[1]))
	lcnt+=1

  
word_set=set(all_words)
i=0
word_idx = dict((c, i + 1) for i, c in enumerate(word_set,start = -1))
reverse_word_map = dict(map(reversed, word_idx.items()))
data=encode_sentences(sents)

## Split
X_train, X_test, Y_train, Y_test = train_test_split(data, labels)
x_train_ids=X_train[:,-1]
x_test_ids=X_test[:,-1]
x_train=X_train[:,:-1]
x_test=X_test[:,:-1]

## Tsetlin machine parameteres
NUM_CLAUSES=40
T=15
s=3.9
TRAIN_EPOCHS=4

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