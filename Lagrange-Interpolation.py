import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def interpolate(f: list, xi, n: int):
    result = 0.0
    for i in range(n):
        term = f[i].y
        for j in range(n):
            if j != i:
                if f[i].x == f[j].x:
                    return [False, result]
                term *= ((xi - f[j].x) / (f[i].x - f[j].x))
        result += term
    return [True, result]


def drawGraph(f, n):
    x = np.array([f[i].x for i in range(n)])
    y = np.array([f[i].y for i in range(n)])
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    n = int(input())
    f = np.zeros(shape=n, dtype=Point)
    for i in range(n):
        x, y = map(int, input().split())
        f[i] = Point(x, y)
    value = int(input())
    drawGraph(f, n)
    status, result = interpolate(f, value, n)
    if status:
        print("Value is : %.4f" % result)
    else:
        print("Can't be calculated")

'''
4
0 2
1 3
2 12
5 147
3
'''
