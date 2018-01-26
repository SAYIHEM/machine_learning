from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

from machines import Machine

__all__ = ["XORMachine"]


class XORMachine(Machine):

    def __init__(self):
        super().__init__()

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        model = Sequential()
        model.add(Dense(8, input_dim=2))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        sgd = SGD(lr=0.1)
        model.compile(loss='binary_crossentropy', optimizer=sgd)

        model.fit(X, y, batch_size=1, nb_epoch=1000)
        print(model.predict_proba(X))
        """
        [[ 0.0033028 ]
         [ 0.99581173]
         [ 0.99530098]
         [ 0.00564186]]
        """

        plt.plot(y)
        plt.plot(model.predict(X))
        plt.show()


    def init(self):
        pass

    def evaluate(self):
        pass