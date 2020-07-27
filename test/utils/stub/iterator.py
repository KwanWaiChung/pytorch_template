import math


class DummyIterator:
    def __init__(self, X, y, batch_size=5):
        self.batch_size = batch_size
        self.X = X
        self.y = y

    def __iter__(self):
        for i in range(0, self.X.shape[0], self.batch_size):
            yield self.X[i : i + self.batch_size], self.y[
                i : i + self.batch_size
            ]

    def __len__(self):
        return math.ceil(self.X.shape[0] / self.batch_size)
