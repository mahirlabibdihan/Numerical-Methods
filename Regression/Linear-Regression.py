import numpy as np
import matplotlib.pyplot as plt


def linear(x, y):
    n = np.size(x)
    p = np.sum(x*x)
    q = np.sum(x)
    r = np.sum(x*y)
    s = np.sum(y)
    a_1 = (n*r-q*s) / (n*p - q*q)
    a_0 = (p*s-q*r) / (n*p - q*q)
    return (a_0, a_1)


def drawGraph(x, y, y_pred):
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()


def main():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    a_0, a_1 = linear(x, y)
    drawGraph(x, y, a_0 + a_1*x)


if __name__ == "__main__":
    main()
