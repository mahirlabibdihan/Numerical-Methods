import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def linear(x, y):
    n = np.size(x)
    p = np.sum(x*x)
    q = np.sum(x)
    r = np.sum(x*y)
    s = np.sum(y)
    a_1 = (n*r-q*s) / (n*p - q*q)
    a_0 = (p*s-q*r) / (n*p - q*q)
    data = {'x': np.append(x, q), 'y': np.append(
        y, s), 'x^2': np.append(x*x, p), 'xy': np.append(x*y, r)}
    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df)+1)
    print(df)
    return (a_0, a_1)


def drawGraph(x, y, y_pred):
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()


def main():
    x = np.array([0.01, 0.03, 0.05, 0.07, 0.09,
                 0.11, 0.13, 0.15, 0.17, 0.19, 0.21])
    y = np.array([1.03, 1.06, 1.38, 2.09, 3.54,
                 6.41, 12.6, 22.1, 39.05, 65.32, 99.78])
    a_0, a_1 = linear(np.log(x), np.log(y))
    a = np.exp(a_0)
    b = a_1
    drawGraph(x, y, a * (x**b))


if __name__ == "__main__":
    main()
