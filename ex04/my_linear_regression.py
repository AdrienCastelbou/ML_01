import numpy as np

class MyLinearRegression:
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas.astype(float)

    def gradient_(self, x, y):
        try:
            l = len(x)
            x = np.hstack((np.ones((x.shape[0], 1)), x))
            nabla_J = x.T.dot(x.dot(self.thetas) - y) / l
            return nabla_J
        except:
            return None

    def fit_(self, x, y):
        try:
            for i in range(self.max_iter):
                nabla_J = self.gradient_(x, y)
                self.thetas[0] = self.thetas[0] - self.alpha * nabla_J[0]
                self.thetas[1] = self.thetas[1] - self.alpha * nabla_J[1]
            return self.thetas
        except:
            return None
    
    def predict_(self, x):
        try:
            m = np.hstack((np.ones((x.shape[0], 1)), x))
            return m.dot(self.thetas)
        except:
            return None

    def loss_elem_(self, y, y_hat):
        try:
            return (y_hat - y) ** 2
        except:
            return None
    
    def loss_(self, y, y_hat):
        try:
            return float(1 / (2 * y.shape[0]) * (y_hat - y).T.dot(y_hat - y))
        except:
            return None
    
    def mse_(y, y_hat):
        try:
            return float(1 / (y.shape[0]) * (y_hat - y).T.dot(y_hat - y))
        except:
            return None
