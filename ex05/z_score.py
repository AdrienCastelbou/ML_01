import numpy as np

def zscore(x):
    if type(x) != np.ndarray or len(x) == 0:
        return None
    normalized = np.zeros(x.shape)
    mean_ = np.mean(x)
    std_ = np.std(x)
    for i in range(len(x)):
        normalized[i] = (x[i] - mean_) / std_
    print(normalized)

def main_test():
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    zscore(X)
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    zscore(Y)

if __name__ == "__main__":
    main_test()