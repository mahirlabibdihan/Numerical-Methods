import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def productTerm(i, value, x):
    pro = 1
    for j in range(i):
        pro *= (value - x[j])
    return pro


def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            if x[j] == x[i + j]:
                return [False, y]
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j]))
    return [True, y]


def applyFormula(value, x, y, n):
    sum = 0
    for i in range(n):
        sum += (productTerm(i, value, x) * y[0][i])
    return [True, sum]


def printDiffTable(y, n):
    for i in range(n):
        for j in range(n - i):
            print("%10.4f" % y[i][j], end="  ")
        print("")


def interpolate(f, value, n, show=True):
    x = np.zeros(shape=n)
    y = np.zeros(shape=(n, n))
    for i in range(n):
        x[i] = f[i].x
        y[i][0] = f[i].y
    status, y = dividedDiffTable(x, y, n)
    if not status:
        return [False, 0]
    if show:
        printDiffTable(y, n)
    return applyFormula(value, x, y, n)


def drawGraph(f, n):
    x = np.array([f[i].x for i in range(n)])
    y = np.array([f[i].y for i in range(n)])
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def main():
    n = int(input())
    f = np.zeros(shape=n, dtype=Point)
    for i in range(n):
        x, y = map(float, input().split())
        f[i] = Point(x, y)
    value = float(input())
    drawGraph(f, n)
    status, result = interpolate(f, value, n)
    if status:
        print("Value is : %.4f" % result)
    else:
        print("Can't be calculated")


if __name__ == "__main__":
    main()

'''
4
10 227.04
15 362.78
20 517.35
22.5 602.97
16
'''
