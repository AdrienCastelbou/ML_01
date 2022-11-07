import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt


def show_datas_with_predict(x, y, y_hat):
    plt.scatter(x, y, label="Strue(pills)")
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.plot(x, y_hat, color="green", linestyle = 'dashed', marker="X", label = "Spredict(pills)")
    plt.grid()
    plt.legend()
    plt.show()

def J_loss_function_evolution(lr, x, y):
    thetas0 = np.linspace(lr.thetas[0], lr.thetas[0] + 10,6)
    #thetas1 = np.linspace(thetas[1],thetas[1] + 10,6)
    y_hat = lr.predict_(x)
    print(y_hat, thetas0)
    loss_ = lr.loss_(y, y_hat)
    print(loss_)

def main(): 
    data = pd.read_csv("are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1,1)
    Yscore = np.array(data["Score"]).reshape(-1,1)
    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    J_loss_function_evolution(linear_model1, Xpill, Yscore)
    return 
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    show_datas_with_predict(Xpill, Yscore, Y_model1)
    Y_model2 = linear_model2.predict_(Xpill)
    show_datas_with_predict(Xpill, Yscore, Y_model2)
    print(MyLR.mse_(Yscore, Y_model1))
    # 57.60304285714282
    print(MyLR.mse_(Yscore, Y_model2))



if __name__ == "__main__":
    main()