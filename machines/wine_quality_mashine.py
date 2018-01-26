import keras
import numpy
from keras import Sequential
from keras.callbacks import Callback
from keras.layers import Dense
from matplotlib import pyplot as plt
from IPython.display import clear_output


from machines import Machine



__all__ = ["WineQuality"]


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show()


plot = PlotLearning()

class UpdatePlotCallback(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        m = self.model

        a =logs.get('loss')
        b=logs.get('val_loss') # TODO: Fix None error

        if a is not None and b is not None:
            plt.plot(logs.get('loss'))
            plt.plot(logs.get('val_loss'))
            plt.show()

class WineQuality(Machine):


    def __init__(self, data):
        super().__init__(data)

    def init(self):
        # fix random seed for reproducibility
        seed = 7
        numpy.random.seed(seed)

    def train(self):

        # split into input (X) and output (Y) variables
        num_in = 4
        X = self.data[:, 0:num_in]
        Y = self.data[:, 11]

        # create model
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=num_in, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        X_train = None
        Y_train = None

        X_test = None
        Y_test = None
        #self.model.fit(X, Y, epochs=500, batch_size=10, validation_data=(X_test, Y_test),callbacks=[plot])
        #self.model.fit(X_train, Y_train, epochs=500,batch_size=10,steps_per_epoch=10,verbose=0)

        update_plot = UpdatePlotCallback()
        history = self.model.fit(X, Y, epochs=50, batch_size=10, callbacks=[update_plot])


        # calculate predictions
        #predictions = self.model.predict(X)



    def evaluate(self):

        raise NotImplemented
        # evaluate the model
        scores = self.model.evaluate(self.X, self.Y)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
