import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import pandas as pd


def f(x):
    return x**3 - 0.165*x**2 + 3.993*10**(-4)


def fd(x):
    return 3*x**2 - 0.33*x


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


def table(init, eS, maxIter):
    eA = np.zeros(maxIter+1)
    x = np.zeros(maxIter+1)
    x[0] = init
    for i in range(0, maxIter):
        if fd(x[i]) == 0:
            print("Error: Slope of current point is 0.")
            sys.exit()
        x[i+1] = x[i] - (f(x[i])/fd(x[i]))
        if x[i+1] == 0:
            eA[i+1] = abs((x[i+1]-x[i])/(x[i+1]+sys.float_info.epsilon))*100
        else:
            eA[i+1] = abs((x[i+1]-x[i])/x[i+1])*100
        if eA[i+1] <= eS:
            break

    data = {'x': x, 'eA': eA}
    df = pd.DataFrame(data)
    print(df)


def drawGraph():
    x = np.array(np.arange(-1, 1, .01))
    y = f(x)
    plt.plot(x, y)
    plt.grid()
    plt.title("f(x)=x^3 - 0.165x^2 + 3.993x10^-4")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def take_input():
    x = float(input("Estimate root: "))
    eS = float(input('Enter max tolerance: '))
    maxIter = int(input("Enter max allowed iteration: "))
    table(x, eS, maxIter)


def main():
    # drawGraph()
    # take_input()
    table(.05, .05, 10)
    # ans = root(x, 0.05, maxIter)
    # if ans == None:
    #     print("No solution exists")
    # else:
    #     print("Solution: ", ans)


if __name__ == "__main__":
    main()
# x^3 - 0.165x^2 + 3.993x10^-4
# 0.14636
# 0.06238
# -0.04374
# tangent in 0 at x=0,0.11
