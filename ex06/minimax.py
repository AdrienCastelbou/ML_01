import numpy as np

def minmax(x):
    try:
        if type(x) != np.ndarray or len(x) == 0:
            return None
        if x.ndim == 1:    
            x = x.reshape(-1, 1)
        if x.shape[0] != 1 and x.shape[1] != 1:
            return None
        minmaxed = np.zeros(x.shape)
        min_ = np.min(x)
        max_ = np.max(x)
        for i in range(len(x)):
            minmaxed[i] = (x[i] - min_) / (max_ - min_)
        return minmaxed
    except:
        return None


def main_test():
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    print(minmax(X))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(minmax(Y))

if __name__ == "__main__":
    main_test()