from machines import XORMachine, WineQuality, Machine


def main():
    pass


if __name__ == "__main__":

    #machine = XORMachine()
    machine = WineQuality("data/winequality-white.csv")
    machine.train()
    #machine.evaluate()