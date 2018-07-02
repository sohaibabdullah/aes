import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



import os
from ... import config
from . import preProc
from . import getEmbed
from ..models import Glove2NN
from . import metric


#initializing parameters
essaySetNumber = 1
maxWords = 50000
maxlen = 350


#importing Excel file containing essays
path = config.dataset_path()
data = pd.read_excel(os.path.join(path, 'training_set_rel3.xls'))
#print(data.head(2))


#Loading essay set number equal to <essaySetNumber> into x and y
x,y= preProc.getEssaySet(data,essaySetNumber)


#Converting words with equvalent numeric dictinary token with maximum length equal to <maxWords>
sequences = preProc.text2token(x,y,max_words=maxWords)


#Make essays into a fixed length sequences with size equla to <maxlen>
sequences = preProc.padSequences(sequences,maxlen,padding='post')
#print(sequences[0])


#Converting labels y into one-hot encoding vector
y,classes = preProc.label21hot(y)
#print(y[0])
#print(np.argmax(y[0]))


#Splitting set into training and test sets
train_x,test_x,train_y,test_y= preProc.ttestSplit(sequences,y,random_state=0,test_size=0.1)


#Loading Glove word embeddings with dimension sieze equal to <embedding_dim>
word2index, embedding_matrix = getEmbed.loadGlove(embedding_dim=50)
#print(embedding_matrix.shape)


hiddenunits = 50
softmaxunits = classes
lr = 0.0001
l2=0.0001
decay = 0.0
loss = 'categorical_crossentropy'
epochs = 20
verbose = 2

y_predicted=Glove2NN.runGlove2NN(train_x, train_y,test_x,test_y,embedding_matrix,maxlen,hiddenunits,softmaxunits,lr,decay,loss,epochs,verbose,l2)

print('QWK:',metric.calculateKappa(y_predicted,test_y))

