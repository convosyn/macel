import numpy as np
import pandas as pd

def func(y_dis):
    y_dis = y_dis.reshape((-1, 1))
    for i in range(y_dis.size):
        if y_dis[i] < 0.10:
            y_dis[i] = 0
        elif y_dis[i] < 0.30:
            y_dis[i] = 1
        elif y_dis[i] < 0.50:
            y_dis[i] = 2
        elif y_dis[i] < 0.70:
            y_dis[i] = 3
        elif y_dis[i] < 0.90:
            y_dis[i] = 4
        else:
            y_dis[i] = 5
    return y_dis

def generate(variables = 5, epsilon_params = 10, epsilon_coef = 20, number_of_inst = 100, bias_term_coef = 5):
    parameters = np.random.rand(variables+1, 1) * 2 * epsilon_params - epsilon_params
    X = np.random.rand(number_of_inst, variables) * 2 * epsilon_coef - epsilon_coef + bias_term_coef
    Xb = np.column_stack([np.ones((number_of_inst, 1)), X])
    y = np.sin(Xb @ parameters)
    y = func(y)
    data = np.column_stack([y, X])
    data = pd.DataFrame(data, columns=range(variables + 1))
    np.savetxt("params", parameters)
    data.to_csv("train_demo.csv", header=None, index=None)

if __name__ == "__main__":
    generate()

