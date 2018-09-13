# -*- coding: utf-8 -*-
from keras.models import Sequential
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

class Model:

    def createModel(self,  
                    X_train, 
                    y_train,
                    input_size = (28,28),
                    hidden_layer_activation = 'relu',
                    output_layer_activation = 'softmax',
                    dropout = 0.2, 
                    optimizer = 'adam',
                    loss = 'sparse_categorical_crossentropy',
                    epochs = 5
                  ):
        model = Sequential()
        # ADD first layer
        #model.add(Conv2D(28, (2,2), input_shape=(*image_size, 1), padding='same', activation='relu'))
        #model.add(MaxPooling2D(pool_size = (2,2)))
        
        #model.add(Conv2D(28, (2,2), padding='same', activation='relu'))
        #model.add(MaxPooling2D(pool_size = (2,2)))
        
        model.add(Flatten())
        
        model.add(Dense(units=512, activation= hidden_layer_activation))
        model.add(Dropout(dropout))
        model.add(Dense(units=32, activation= hidden_layer_activation))
        model.add(Dense(units=10, activation= output_layer_activation))
        
        model.compile(optimizer= optimizer, loss = loss, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs)
       
        self.saveModel(model)
        del model


    
    def saveModel(self, classifier):
        classifier.save('model/myMNITSmodel.h5')
        
    def loadModel(self, modelPath = 'model/myMNITSmodel.h5'):
        model = load_model(modelPath)
        return model