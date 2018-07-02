import numpy as np
np.random.seed(200)
from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense,LSTM,Dropout
from keras import metrics, optimizers, regularizers

def buildGlove2NN(hiddenunits,softmaxunits,embedding_matrix,maxlen,l2):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],output_dim=embedding_matrix.shape[1],input_length=maxlen,
                   weights=[embedding_matrix],trainable=False,name='embedding_layer'))
    model.add(Flatten())
    model.add(Dense(hiddenunits, activation='relu',kernel_regularizer=regularizers.l2(l2),))
    model.add(Dense(softmaxunits, activation='softmax'))
    model.summary()
    return model




def compileGlove2NN(model,train_x, train_y,test_x,test_y,lr,decay,loss,epochs,verbose):
    opt=optimizers.RMSprop(lr=lr, decay=decay)
    model.compile(opt, loss=loss, metrics=['accuracy'])  # Compile the model
    model.fit(train_x, train_y, epochs = epochs, verbose=verbose,validation_data=(test_x,test_y))  # Fit the model

    return model




def runGlove2NN(train_x, train_y,test_x,test_y,embedding_matrix,maxlen,hiddenunits,softmaxunits,lr,decay,loss,epochs,verbose,l2):
    model=buildGlove2NN(hiddenunits,softmaxunits,embedding_matrix,maxlen,l2)
    model=compileGlove2NN(model,train_x, train_y,test_x,test_y,lr,decay,loss,epochs,verbose)

    y_predicted = model.predict(test_x)
    return y_predicted

  
