import numpy as np

def gradient(x, y, theta):
    try:
        if type(x) != np.ndarray or type(theta) != np.ndarray or type(y) != np.ndarray:
            return None
        if theta.ndim == 1:
            theta = theta.reshape(theta.shape[0], -1)
        if not len(x) or theta.shape != (2, 1):
            return None
        if x.ndim == 1:
            x = x.reshape(x.shape[0], -1)
        if x.shape[1] != 1:
            return None
        if y.ndim == 1:
            y = y.reshape(x.shape[0], -1)
        if y.shape[1] != 1:
            return None
        l = len(x)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        nabla_J = x.T.dot(x.dot(theta) - y) / l
        return nabla_J
    except:
        return None

def main_test():
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(gradient(x, y, theta1))
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(gradient(x, y, theta2))

if __name__ == "__main__":
    main_test()