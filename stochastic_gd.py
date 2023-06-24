import numpy as np


class SGDRegressor:

    def __init__(self, learning_rate=0.01, epochs=100):
        self.cof = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, x_train, y_train):
        self.cof = np.ones(x_train.shape[1])
        self.intercept_ = 0

        for i in range(self.epochs):
            for j in range(x_train.shape[1]):
                id = np.random.randint(0, x_train.shape[0])

                y_hat = np.dot(x_train[id], self.cof) + self.intercept_

                intercept_der = -2 * np.mean(y_train[id] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                cofder = -2 * np.dot((y_train[id] - y_hat), x_train[id])
                self.cof = self.cof - (self.lr * cofder)

        print(self.intercept_, self.cof)

    def predict(self, x_test):
        return np.dot(x_test, self.cof) + self.intercept_
