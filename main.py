import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.mnist as mnist
from generateNeuralNetworkModel import Model
import math
class Main:
    
    def readData(self):
        
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        
        #Scaling the MNIST data
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
        return (x_train, y_train),(x_test, y_test)
        
        
    def trainTheModel(self,
                      x_train, 
                      y_train):
        gnnm = Model()
        gnnm.createModel(x_train, y_train)
        
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
        plt.show()
        
        
        
        
main = Main()
(x_train, y_train),(x_test, y_test) = main.readData()
#main.trainTheModel(x_train, y_train)

model = Model().loadModel()

ev = model.evaluate(x_test, y_test)
loss, accuracy = ev[0], ev[1]

image_size = (28, 28)
y_pred = np.array([0] * (len(x_test)))
incorrect = list()
for i in range(len(x_test)):
    y_pred[i] = (np.argmax(max(model.predict(np.reshape(x_test[i], (1, *image_size))))))
    if y_test[i] != y_pred[i]:
        incorrect.append(i)
#SHOW INCORRENCT PREDICTIONS
main.plotImage(data=x_test, incorrect = incorrect)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)