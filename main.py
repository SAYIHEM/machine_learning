from machines import XORMachine, WineQuality, Machine
from matplotlib import pyplot as plt
import numpy as np

def main():
    pass

if __name__ == "__main__":

    #plot_test()

    #machine = XORMachine()
    machine = WineQuality("data/winequality-white.csv")
    machine.train()
    #machine.evaluate()