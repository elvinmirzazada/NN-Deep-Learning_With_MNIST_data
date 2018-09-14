# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
import matplotlib.pyplot as plt
class Model:

    def createModel(self,  
                    X_train, 
                    y_train,
                    X_test,
                    y_test,
                    input_size = (28,28),
                    hidden_layer_activation = 'relu',
                    output_layer_activation = 'softmax',
                    dropout = 0.4, 
                    optimizer = 'adam',
                    loss = 'sparse_categorical_crossentropy',
                    epochs = 5,
                    batch_size=64
                  ):
        model = Sequential()
        # ADD first layer
        model.add(Conv2D(32, (3, 3), input_shape=(*input_size, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
#        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#        model.add(MaxPooling2D(pool_size = (2,2)))
        
        model.add(Flatten())
        
        model.add(Dense(units=512, activation= hidden_layer_activation))
        model.add(Dropout(dropout))
        model.add(Dense(units=32, activation= hidden_layer_activation))
        model.add(Dropout(dropout/2))
        model.add(Dense(units=10, activation= output_layer_activation))
        
        model.compile(optimizer= optimizer, loss = loss, metrics=['accuracy'])
        model_history = model.fit(X_train, 
                                  y_train, 
                                  epochs=epochs,
                                  batch_size=batch_size, 
                                  validation_data=(X_test, y_test))
        
        
        accuracy = model_history.history['acc']
        val_accuracy = model_history.history['val_acc']
        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        self.saveModel(model)
        del model


    
    def saveModel(self, classifier):
        classifier.save('model/myMNITSmodel.h5')
        
    def loadModel(self, modelPath = 'model/myMNITSmodel.h5'):
        model = load_model(modelPath)
        return model