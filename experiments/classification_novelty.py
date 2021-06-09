import sys
import os 
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as panda
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
import numpy as np
from time import time
from sklearn import preprocessing


if len(sys.argv)<2:
	print('Not enough arguments, should be ','python '+sys.argv[0]+' filename')
	print('Check usage in readme.txt')
	exit()

if '20_newsgroups' in sys.argv[1]:
	infile='../data/dt_newsgroup_cuda.csv'
if 'BBC' in sys.argv[1]:
	infile='../data/dt_bbc_cuda.csv'

#Read data
df=panda.read_csv(infile)   #load file from classification_data
X=df.iloc[:,:2]
Y= df.iloc[:,2:]
Y=np.ravel(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=42)

#Scaling the input for MLP
from sklearn.externals.six import StringIO 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test =scaler.transform(X_test)
print("scale",X_train.shape)

#MLP classification
mlp = MLPClassifier(hidden_layer_sizes=(100, 30), max_iter=300, alpha=1e-4,activation='relu', 
                    solver='sgd', verbose=False, tol=1e-4, random_state=42,
                    learning_rate_init=.1)     
mlp.fit(X_train, Y_train)
y_pred_mlp= mlp.predict(X_test)

print("f-score", f1_score(Y_test, y_pred_mlp))
print("Training set score: %f" % mlp.score(X_test, Y_test))