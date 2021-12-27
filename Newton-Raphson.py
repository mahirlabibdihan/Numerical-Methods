import numpy as np
import matplotlib.pyplot as plt
import math
import sys


def f(x):
    # return math.sin(x)
    return x*x*x - 0.165*x*x + 0.0003993
    # return (x-1)**3+0.512
    # return x*x+2
    # return -(x-2)**2+2


def fd(x):
    # return math.cos(x)
    return 3*x*x - 0.33*x
    # return 3*(x-1)**2
    # return -2*(x-2)


def root(x, eS, maxIter):
    for i in range(maxIter):
        if fd(x) == 0:
            print("Error: Slope of current point is 0.")
            sys.exit()
        xNew = x - (f(x)/fd(x))
        if xNew == 0:
            eA = abs((xNew-x)/(xNew+sys.float_info.epsilon))*100
        else:
            eA = abs((xNew-x)/xNew)*100
        # print(x, xNew, eA)
        if eA <= eS:
            # print("Process ended: Relative error is less than", eS)
            return xNew
        x = xNew
    # print("Process ended: Maximum iteration limit exceeded.")
    return


def drawGraph():
    x = np.array(np.arange(-1, 1, .01))
    y = f(x)
    plt.plot(x, y)
    plt.grid()
    plt.title("f(x)=x^3 - 0.165x^2 + 3.993x10^-4")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def main():
    drawGraph()
    x = float(input("Estimate root: "))
    maxIter = int(input("Enter max allowed iteration: "))
    ans = root(x, 0.05, maxIter)
    if ans == None:
        print("No solution exists")
    else:
        print("Solution: ", ans)


if __name__ == "__main__":
    main()
# x^3 - 0.165x^2 + 3.993x10^-4
# 0.14636
# 0.06238
# -0.04374
# tangent in 0 at x=0,0.11
