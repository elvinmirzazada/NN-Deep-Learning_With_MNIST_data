import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.mnist as mnist
from generateNeuralNetworkModel import Model
import math
class Main:
    
    def readData(self):
        
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        
        x_train = np.reshape(x_train, (-1, 28,28, 1))
        x_test =  np.reshape(x_test, (-1, 28,28, 1))
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        #Scaling the MNIST data
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
#        train_Y_one_hot = to_categorical(y_train)
#        test_Y_one_hot = to_categorical(y_test)
#        
        return (x_train, y_train),(x_test, y_test)
        
        
    def trainTheModel(self,
                      x_train, 
                      y_train, 
                      x_test, 
                      y_test):
        gnnm = Model()
        gnnm.createModel(x_train, y_train, x_test, y_test)
        
    def plotImage(self, data, incorrect):
        
        fig=plt.figure(figsize=(8, 8))
        columns = 10
        rows = math.ceil(len(incorrect)/10)
        print(rows)
        for i in range(0, len(incorrect)):
            fig.add_subplot(rows, columns, i+1)
            plt.axis('off')
            plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
            plt.imshow(np.reshape(data[incorrect[i]], (28, 28)))
            plt.margins(x = 1, y=10)
        plt.show()
        
        
        
        
main = Main()
(x_train, y_train),(x_test, y_test) = main.readData()

main.trainTheModel(x_train, y_train, x_test, y_test)

model = Model().loadModel()

ev = model.evaluate(x_test, y_test)
predicted_classes = model.predict(x_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
incorrect = np.where(predicted_classes != y_test)[0]

#SHOW INCORRENCT PREDICTIONS
main.plotImage(data=x_test, incorrect = incorrect)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)





















