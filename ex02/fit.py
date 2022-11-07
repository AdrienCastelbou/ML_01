import numpy as np

def predict(x, theta) -> np.array:
    try:
        if type(x) != np.ndarray or type(theta) != np.ndarray:
            return None
        if theta.ndim == 1:
            theta = theta.reshape(theta.shape[0], -1)
        if not len(x) or theta.shape != (2, 1):
            return None
        if x.ndim == 1:
            x = x.reshape(x.shape[0], -1)
        if x.shape[1] != 1:
            return None
        m = np.hstack((np.ones((x.shape[0], 1)), x))
        return m.dot(theta)
    except:
        None

def gradient(x, y, theta):
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

def fit_(x, y, theta, alpha, max_iter):
    try:
        if type(alpha) != float or type(max_iter) != int:
            return None
        new_theta = np.array([float(theta[0]), float(theta[1])]).reshape(-1, 1) 
        for i in range(max_iter):
            nabla_J = gradient(x, y, new_theta)
            new_theta[0] = new_theta[0] - alpha * nabla_J[0]
            new_theta[1] = new_theta[1] - alpha * nabla_J[1]
        return new_theta
    except:
        return None

def main_test():
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta= np.array([1, 1]).reshape((-1, 1))

    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    print("Predict after imporvement : ", predict(x, theta1))
    print("Predict with largest dataset")
    x = np.random.rand(20,1).reshape((-1, 1))
    y = np.random.rand(20, 1).reshape((-1, 1))
    print("Target : ", y)
    print("Predict before improvement : ", predict(x, theta1))
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print("new theta = ", theta1)
    print("Predict after imporvement : ", predict(x, theta1))


if __name__ == "__main__":
    main_test()