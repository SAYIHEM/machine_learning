from machines import XORMachine, WineQuality, Machine
from matplotlib import pyplot as plt
import numpy as np

def main():
    pass


def plot_test():
    plt.axis([0, 10, 0, 1])
    plt.ion()

    for i in range(10):
        y = np.random.random()
        plt.scatter(i, y)
        plt.pause(0.05)

    while True:
        plt.pause(0.05)


if __name__ == "__main__":

    #plot_test()

    #machine = XORMachine()
    machine = WineQuality("data/winequality-white.csv")
    machine.train()
    #machine.evaluate()