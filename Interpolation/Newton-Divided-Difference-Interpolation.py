import numpy as np
import matplotlib.pyplot as plt

# More points doesn't mean less error


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


def nearestPoints(f, n, value):
    # We need to choose n data points that are closest to 'value' , that also bracket 'value' to evaluate it.
    m = len(f)
    if m < n:
        return [False, None, None]
    x = np.zeros(shape=n)
    y = np.zeros(shape=(n, n))
    temp1 = {}
    for i in range(m):
        temp1[f[i].x] = abs(f[i].x-value)
    d = sorted(temp1, key=temp1.get)    # Sort dictionary

    hasLowerPoints = False
    hasUpperPoints = False
    for i in range(n):
        x[i] = d[i]
        if(d[i] >= value):
            hasUpperPoints = True
        if(d[i] <= value):
            hasLowerPoints = True
    # Checking if choosen points have bracket 'value' or not
    if not hasLowerPoints:
        for i in range(m):
            if(d[i] <= value):
                x[n-1] = d[i]
                hasLowerPoints = True
                break
    if not hasUpperPoints:
        for i in range(m):
            if(d[i] >= value):
                x[n-1] = d[i]
                hasUpperPoints = True
                break
    if not hasLowerPoints or not hasUpperPoints:
        return [False, None, None]
    x = np.sort(x)
    for i in range(n):
        for j in range(m):
            if(f[j].x == x[i]):
                y[i][0] = f[j].y
                break
    return [True, x, y]


def interpolate(f, value, n, show=True):
    status, x, y = nearestPoints(f, n, value)
    if not status:
        return [False, 0]
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
    m = int(input())
    f = np.zeros(shape=m, dtype=Point)
    for i in range(m):
        x, y = map(float, input().split())
        f[i] = Point(x, y)
    value = float(input())
    drawGraph(f, m)
    status, result = interpolate(f, value, 4)
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
