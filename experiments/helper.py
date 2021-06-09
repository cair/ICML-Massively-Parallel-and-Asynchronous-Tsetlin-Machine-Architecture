from nltk.corpus import stopwords
import numpy as np
import nltk
import string
nltk.download('stopwords')
np.random.seed(400) 
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

# function to preprocess the words list to remove punctuations
def preprocess(words):
    # we'll make use of python's translate function,that maps one set of characters to another
    # we create an empty mapping table, the third argument allows us to list all of the characters
    # to remove during the translation process
    # first we will try to filter out some  unnecessary data like tabs
    table = str.maketrans('', '', '\t')
    words = [word.translate(table) for word in words]
    punctuations = (string.punctuation).replace("'", "")
    # the character: ' appears in a lot of stopwords and changes meaning of words if removed
    # hence it is removed from the list of symbols that are to be discarded from the documents
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in words]
    # some white spaces may be added to the list of words, due to the translate function & nature of our documents
    # we remove them belowr
    words = [str for str in stripped_words if str]
    # some words are quoted in the documents & as we have not removed ' to maintain the integrity of some stopwords
    # we try to unquote such words below
    p_words = []
    for word in words:
        if (word[0] and word[len(word) - 1] == "'"):
            word = word[1:len(word) - 1]
        elif (word[0] == "'"):
            word = word[1:len(word)]
        else:
            word = word
        p_words.append(word)
    words = p_words.copy()
    # we will also remove just-numeric strings as they do not have any significant meaning in text classification
    words = [word for word in words if not word.isdigit()]
    # we will also remove single character strings
    words = [word for word in words if not len(word) == 1]
    # after removal of so many characters it may happen that some strings have become blank, we remove those
    words = [str for str in words if str]
    # we also normalize the cases of our words
    words = [word.lower() for word in words]
    # we try to remove words with only 2 characters
    words = [word for word in words if len(word) > 2]
    return words

def remove_stopwords(words):
    words = [word for word in words if not word in stopwords]
    return words

# function to convert a sentence into list of words
def tokenize_sentence(line):
    words= line[0:len(line)-1].strip().split(" ")
    words = preprocess(words)
    words = remove_stopwords(words)
    return words

#function to remove metadata
def remove_metadata(lines):
    start=0
    for i in range(len(lines)):
        if(lines[i] == '\n'):
            start = i+1
            break
    new_lines = lines[start:]
    return new_lines


# function to convert a document into list of words
def tokenize(path):
    # load document as a list of lines
    f = open(path, encoding="utf8", errors='ignore')
    text_lines = f.readlines()
    # removing the meta-data at the top of each document
    text_lines = remove_metadata(text_lines)
    # initiazing an array to hold all the words in a document
    doc_words = []
    # traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
    for line in text_lines:
        doc_words.append(tokenize_sentence(line))
    return doc_words

#a simple helper function to convert a 2D array to 1D, without using numpy

def flatten(list):
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list
list_of_words = []
list_of_words_novel = []
list_of_words_word_idx=[]
list_of_words_test=[]