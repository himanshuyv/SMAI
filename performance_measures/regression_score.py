import numpy as np

class Scores:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
        self.mse = np.mean((y_pred - y_test) ** 2)