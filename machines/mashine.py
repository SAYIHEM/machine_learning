from abc import ABC, abstractmethod, ABCMeta
from pathlib import Path

import numpy

__all__ = ["Machine"]


class Machine(object):

    __metaclass__ = ABCMeta

    data_file = None
    data = None
    model = None

    def __init__(self, data=None):
        self.load_data(data)

    def load_data(self, csv_file):
        if isinstance(csv_file, str):
            csv_file = Path(csv_file)
        if not isinstance(csv_file, Path):
            raise AttributeError("'data_file' is no instance of string or path. data_file: "+ csv_file)

        if not csv_file.is_file():
            raise FileNotFoundError()

        self.data_file = csv_file

        self.data = numpy.loadtxt(str(self.data_file), delimiter=",")

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def evaluate(self): pass



