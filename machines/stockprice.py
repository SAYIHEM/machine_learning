import keras
import numpy
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from pprint import pprint as pp
from IPython.display import clear_output

from machines import Machine


__all__ = ["WineQuality"]


class UpdatePlotCallback(Callback):

    epochs = []
    losses = []

    def __init__(self):
        super().__init__()



    def on_train_begin(self, logs=None):
        #plt.axis([0, 1, 0, ])
        plt.ion()

    def on_epoch_end(self, epoch, logs={}):

        self.losses.append(logs.get('loss'))
        self.epochs.append(epoch)

        val_loss = logs.get('val_loss') # TODO: Fix None error

        plt.plot(self.epochs, self.losses, linestyle='solid')
        plt.pause(0.001)
        plt.show()
        plt.gcf().clear()

    def on_train_end(self, logs=None):
        plt.close()

        # if a is not None and b is not None:
        #     plt.plot(logs.get('loss'))
        #     plt.plot(logs.get('val_loss'))
        #     plt.show()

class Stockpredictor(Machine):


    def __init__(self, data):
        super().__init__(data)

    def init(self):
        # fix random seed for reproducibility
        seed = 7
        numpy.random.seed(seed)

        plt.ion()


    def train(self):

        # split into input (X) and output (Y) variables
        num_in = 3
        X = self.data[:, 0:num_in]
        Y = self.data[:, 3]

        X = X*0.0001
        Y = Y*0.0001

        pp(X)
        pp(Y)

        #X = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        #Y = numpy.array([[0], [1], [1], [0]])
        # create model
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=num_in, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(6, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile model
        self.model.compile(loss='binary_crossentropy',
                           optimizer=SGD(lr=0.01),
                           metrics=['accuracy'])

        # Fit the model
        update_plot = UpdatePlotCallback()
        epochs = 500
        history = self.model.fit(X, Y, epochs=epochs, batch_size=3, callbacks=[update_plot])

        #print performance
        scores = self.model.evaluate(X, Y, verbose=0)
        print('Performance accuracy (%): ' + str(scores))
        # calculate predictionslen
        test = numpy.array([[8.3,0.42,0.62,19.25,0.04,41,172,1.0002,2.98,0.67,9.7]]) # 5
        predictions = self.model.predict(X)
        plt.plot(predictions)
        plt.plot(Y)
        plt.show(block=True)

    def evaluate(self):
        scores = self.model.evaluate(self.X, self.Y, verbose=0)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
