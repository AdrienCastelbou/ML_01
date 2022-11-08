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
    thetas0 = np.linspace(86, 92, 6)
    thetas1 = np.linspace(-14, -4, 100)
    ax = plt.gca()
    ax.set_ylim([0, 140])
    for t0 in thetas0:
        cost_values = []
        for t1 in thetas1:
            lr.thetas = np.array([[t0], [t1]])
            y_hat = lr.predict_(x)
            cost_values.append(lr.loss_(y, y_hat))
        plt.plot(thetas1,cost_values, label=f"J(theta0={t0}, theta1")
    plt.xlabel("theta1")
    plt.ylabel("Cost function J(theta0, theta1)")
    plt.legend()
    plt.grid()
    plt.show()



def main(): 
    data = pd.read_csv("are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1,1)
    Yscore = np.array(data["Score"]).reshape(-1,1)
    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    show_datas_with_predict(Xpill, Yscore, Y_model1)
    Y_model2 = linear_model2.predict_(Xpill)
    show_datas_with_predict(Xpill, Yscore, Y_model2)
    J_loss_function_evolution(lr = linear_model1,x= Xpill,y= Yscore)
    print(linear_model1.mse_(Yscore, Y_model1))
    print(linear_model1.mse_(Yscore, Y_model2))



if __name__ == "__main__":
    main()