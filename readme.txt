PyTsetlinMachineCUDA - Massively Parallel and Asynchronous Architecture for Logic-based AI
Using logical clauses to represent patterns, Tsetlin machines (https://arxiv.org/abs/1804.01508) have obtained competitive performance in terms of accuracy, memory footprint, energy, and learning speed on several benchmarks (image classification, regression and natural language understanding). In the parallel and asynchronous architecture implemented here, each clause runs in its own thread for massive parallelism. The clauses access the training examples simultaneously, updating themselves and local voting tallies in parallel. A team of Tsetlin Automata composes each clause. The Tsetlin Automata thus drive the entire learning process. These are rewarded/penalized according to three local rules that optimize global behaviour.
There is no synchronization among the clause threads, apart from atomic adds to the local voting tallies. Hence, the speed up!

This document is a how-to for running the experiments detailed in the paper.
All data required is found in data/., all experiment code is found in experiments/.

================
Requirements
================
Python 3.7.x, https://www.python.org/
Numpy, http://www.numpy.org/
PyCUDA, https://documen.tician.de/pycuda/
Scikit-learn, https://scikit-learn.org/
NLTK, https://www.nltk.org/
Keras, https://keras.io/
Tensorflow, https://www.tensorflow.org/
_______________________________________________________________

================
Installation
================

python setup.py build && python setup.py install
_______________________________________________________________

================
Experiments
================

================
Sentiment Analysis
================
$python classification_imdbsentiment.py

Output:
-------
Producing bit representation...
Selecting features...

Accuracy over 100 epochs:

#1 Accuracy: 85.15% Training: 18.19s Testing: 6.61s
#2 Accuracy: 87.10% Training: 13.54s Testing: 4.71s
#3 Accuracy: 87.72% Training: 13.12s Testing: 4.78s
...

#98 Accuracy: 89.47% Training: 10.49s Testing: 5.82s
#99 Accuracy: 89.70% Training: 10.44s Testing: 5.83s
#100 Accuracy: 89.57% Training: 10.41s Testing: 5.84s

_______________________________________________________________

================
Regression
================
$python regression_demo.py ../data/Energy.txt

or

$python regression_demo.py ../data/AnnualReturn.txt

________________________________________________________________

================
Word Sense Disambiguation
================
$python wordsensedisambiguation.py ../data/apple.csv

or

$python wordsensedisambiguation.py ../data/java.csv

________________________________________________________________

================
Semantic Relation Analysis
================
$python classification_causalrelation.py

________________________________________________________________

================
Novelty Detection
================
Score calculation using TM:

$python noveltyscore.py ../data/BBC

Then, classification using MLP on above generated scores:

$python classification_novelty.py ../data/BBC


or
$python noveltyscore.py ../data/20_newsgroups
$python classification_novelty.py ../data/20_newsgroups

---------------------------------------------------------
---------------------------------------------------------

Original Data Sources
================
Stock portfolio data for Regression from : https://archive.ics.uci.edu/ml/datasets/Stock+portfolio+performance [Annual Return only]
Energy Performance data for Regression from : https://archive.ics.uci.edu/ml/datasets/Energy+efficiency [Heating Load only]
Word Sense Disambiguation Data from https://github.com/danlou/bert-disambiguation/tree/master/data/CoarseWSD-20_balanced [Apple and Java only]
20 Newsgroups Data for Novelty Detection from http://qwone.com/~jason/20Newsgroups/ [comp.graphics, talk.politics.guns and rec.sport.baseball only]
SemEval2010_task8 Data for Semantic Relation Analysis from http://semeval2.fbk.eu/semeval2.php?location=tasks [Cause-Effect only]


----FIN----