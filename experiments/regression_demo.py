from PyTsetlinMachineCUDA.tm import RegressionTsetlinMachine
import numpy as np
from time import time
import sys

if len(sys.argv)<2:
    print('Not enough arguments, should be ','python '+sys.argv[0]+' filename')
    print('Check usage in readme.txt')
    exit()

df = np.loadtxt(sys.argv[1]).astype(dtype=np.float32)

####----Binarization----#####
NOofThresholds = 0
for i in range(len(df[0])-1):
    uniqueValues = list(set(df[:,i]))
    NOofThresholds = NOofThresholds + len(uniqueValues)

NewData = np.zeros((len(df), NOofThresholds+1))

m = -1
for i in range(len(df[0])-1):
    uniqueValues = list(set(df[:,i]))
    uniqueValues.sort() 
    NOofuniqueValues = len(uniqueValues)
    for j in range(NOofuniqueValues):
        m += 1
        for k in range(len(df)):
            if df[k,i] <= uniqueValues[j]:
                NewData[k,m] = 1
            else:
                NewData[k,m] = 0
                
NewData[:,NOofThresholds] = df[:,len(df[0])-1]

## Split
NOofTestingSamples = len(NewData)*20//100
NOofTrainingSamples = len(NewData)-NOofTestingSamples
X_train = NewData[0:NOofTrainingSamples,0:len(NewData[0])-1].astype(dtype=np.int32)
Y_train = NewData[0:NOofTrainingSamples,len(NewData[0])-1:len(NewData[0])].flatten().astype(dtype=np.float32)

X_test = NewData[NOofTrainingSamples:NOofTrainingSamples+NOofTestingSamples,0:len(NewData[0])-1].astype(dtype=np.int32)
X_test.tolist()
Y_test = NewData[NOofTrainingSamples:NOofTrainingSamples+NOofTestingSamples,len(NewData[0])-1:len(NewData[0])].flatten().astype(dtype=np.float32) 


##Tsetlin machine parameteres
if 'bikeS' in sys.argv[1]:
    s = 1.5
if 'AnnualReturn' in sys.argv[1]:
    s = 2
number_of_clauses = T = 1280

#passing parameters to tsetlin machine
tsetlin_machine = RegressionTsetlinMachine(number_of_clauses, T, s, max_weight=1)

#start training and testing
print("\nMAE over 100 epochs:\n")
for epo in range(100):
    start = time()
    tsetlin_machine.fit(X_train, Y_train, epochs=1, incremental=True)
    stop = time()
    pred=tsetlin_machine.predict(X_test)
    MAE = abs(pred - Y_test).mean()
    print("#%d MAE:%.2f%% (%.2fs)",epo+1,MAE, stop - start)
    