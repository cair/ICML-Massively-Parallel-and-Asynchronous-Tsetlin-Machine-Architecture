import numpy as np
import pandas as pd
import sys
from string import punctuation
import pandas as pd
from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
from time import time 
from sklearn import metrics
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer 
from nltk import FreqDist 
from nltk.tokenize import RegexpTokenizer
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
tokenizerR = RegexpTokenizer(r'\w+')

if len(sys.argv)<2:
    print('Not enough arguments, should be ','python '+sys.argv[0]+' filename')
    print('Check usage in readme.txt')
    exit()
#%%%%%%%%%%%%%Read the WSD Dataset %%%%%%%%%%%%%%%%%
dfApple = pd.read_csv(sys.argv[1])
appleLoc = dfApple.iloc[:,0:1].values
appleText = dfApple.iloc[:,1:2].values
appleLabel = dfApple.iloc[:,2:3].values
appleLabel = np.reshape(appleLabel, len(appleLabel))
appleLoc = np.reshape(appleLoc, len(appleLoc))

#Data Preprocessing
def prepreocess(data):
    input_data=[]
    vocab   = []
    for i in data:
        for j in i:
            j = j.lower()
            j = j.replace("\n", "")
            j = j.replace('n\'t', 'not')
            j = j.replace('\'ve', 'have')
            j = j.replace('\'ll', 'will')
            j = j.replace('\'re', 'are')
            j = j.replace('\'m', 'am')
            j = j.replace('/', ' / ')
            j = j.replace('-', ' ')
            j = j.replace('!', ' ')
            j = j.replace('?', ' ')
            j = j.replace('+', ' ')
            j = j.replace('*', ' ')
            while "  " in j:
                j = j.replace('  ', ' ')
            while ",," in j:
                j = j.replace(',,', ',')
            j = j.strip()
            j = j.strip('.')
            j = j.strip()

            temp1 = tokenizerR.tokenize(j)
            temp2 = [x for x in temp1 if not x.isdigit()]
            temp3 = [w for w in temp2 if not w in stop_words]

            # Use PorterStemmet to stem the words in the target 	
            ps = PorterStemmer()
            temp4 = []
            for m in temp3:
                temp_temp =ps.stem(m)
                temp4.append(temp_temp)
            input_data.append(temp4)

            #Append all the words into a list
            for n in temp4:
                vocab.append(n)
 
    return vocab, input_data

#Create a Bag of Words (BOW) for the input sentence and the target word.
def binarization_text(data4,full_token_fil):
    feature_set = np.zeros([len(data4), len(full_token_fil)], dtype=np.uint8)
    
    tnum=0
    for t in data4:
        for w in t:
            if (w in full_token_fil):
                idx = full_token_fil.index(w)
                feature_set[tnum][idx] = 1
        tnum += 1
    return feature_set


vocabApple, inputApple = prepreocess(appleText)  #total words and the input

#Use FreqDist to get most common 3000 words
fdist1 = FreqDist(vocabApple)
tokens1 = fdist1.most_common(3000)

#Create a vocabulary list
full_token_fil = []
for i in tokens1:
    full_token_fil.append(i[0])

dataApple = binarization_text(inputApple,full_token_fil)


#Split training and testing samples
X_train = dataApple[0:1784,:]   #for java dataset change 1784 to 3726
X_test = dataApple[1784:,:]     #for java dataset change 1784 to 3726
Y_train = appleLabel[0:1784]    #for java dataset change 1784 to 3726
Y_test = appleLabel[1784:]      #for java dataset change 1784 to 3726

print(X_test.shape, Y_test.shape)
## Tsetlin machine parameteres
NUM_CLAUSES=300  
T= 50    
s=5.0 

#passing parameters to tsetlin machine
tm =  MultiClassTsetlinMachine(NUM_CLAUSES, T, s)

#start training and testing
print("\nAccuracy over 100 epochs:\n")
for epo in range(100):
    start = time()
    start = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop = time()
    compute= stop-start
    result = 100*(tm.predict(X_train) == Y_train).mean()
    print("#%d Training Accuracy: %.2f%% (%.2fs)" % (epo+1, result, stop-start))
    start = time()
    pred = tm.predict(X_test)
    stop = time()
    result_test = 100*((pred) == Y_test).mean()
    print("#%d Testing Accuracy: %.2f%% (%.2fs)" % (epo+1, result_test, stop-start))
    f1= metrics.f1_score(Y_test, pred, average='macro')
    print("#%d F1-Score: %.2f%%" % (epo+1,f1*100))
