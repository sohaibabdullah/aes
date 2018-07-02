import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split


def getEssaySet(data,setnum):
    df = data[data['essay_set']==setnum]
    x = df['essay']
    y = df['domain1_score']
    return x,y


def text2token(x,y,max_words):
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x)
    return sequences



def label21hot(y):
    y = np.reshape(y,(y.shape[0],1))
    y=y.astype(int)
    y=np_utils.to_categorical(y)
    return y, y.shape[1]


def padSequences(sequences,maxlen,padding='post'):
    sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')
    return sequences


def ttestSplit(sequences,y,random_state=0,test_size=0.1):
    train_x,test_x,train_y,test_y= train_test_split(sequences,y,random_state=random_state,test_size=test_size)
    train_x = np.asarray(train_x)
    test_x = np.asarray(test_x)
    train_y = np.asarray(train_y)
    test_y=np.asarray(test_y)
    return train_x,test_x,train_y,test_y
