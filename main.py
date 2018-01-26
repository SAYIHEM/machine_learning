from machines import XORMachine, WineQuality, Machine
from matplotlib import pyplot as plt
import numpy as np

from machines.stockprice import Stockpredictor


def main():
    pass

if __name__ == "__main__":

    #plot_test()

    #machine = XORMachine()
    machine = Stockpredictor("data/stocks/GOOGL.csv")
    machine.train()


    #machine.evaluate()