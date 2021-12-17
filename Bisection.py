import numpy as np
import matplotlib.pyplot as plt
import sys


def f(x):
    return x-1
    # return x*x*x - 0.18*x*x + 0.0004752

# xL>xU -> Invalid
# xL<xU<root -> Invalid
# root<xL<sU -> Invalid
# xL<xU<=root -> return xU
# root<=xL<xU -> return xL
# xL<=root<xU -> return xL
# xL<root<=xU -> return xU
# xL<=root<=xU -> return xL
# xL<root<xU -> root(xL, xU, eS, maxIter)


def root(xL, xU, eS, maxIter):
    if xL > xU:
        print("Error: Lower bound is greater than upper bound.")
        sys.exit()
    if f(xL)*f(xU) > 0:
        print("Error: Function doesn't changes sign on given bounds.")
        sys.exit()
    xM, xOldM = 0, 0
    for i in range(maxIter):
        xM = (xL+xU)/2
        yL = f(xL)
        yU = f(xU)
        yM = f(xM)
        if yL*yM < 0:
            xU = xM
        elif yU*yM < 0:
            xL = xM
        elif yL*yM == 0:  # yL=0 or yM=0
            if yL == 0:
                return xL
            else:
                return xM
        elif yU*yM == 0:  # yU=0 or yM=0
            if yU == 0:
                return xU
            else:
                return xM
        if i > 0:
            if xM == 0:
                eA = abs((xM-xOldM)/(xM+sys.float_info.epsilon))*100
            else:
                eA = abs((xM-xOldM)/xM)*100
            if eA <= eS:
                return xM
        xOldM = xM
    return xM


def table(xL, xU, eS, maxIter):
    if f(xL)*f(xU) > 0:
        print("Root can't be calculated")
        sys.exit()
    print("Table:")
    print("Iteration No\t\tAbsolute Relative Approximate Error")
    xM, xOldM = 0, 0
    for i in range(maxIter):
        xM = (xL+xU)/2
        yL = f(xL)
        yU = f(xU)
        yM = f(xM)
        if yL*yM < 0:
            xU = xM
        elif yU*yM < 0:
            xL = xM
        elif yL*yM == 0:
            if yL == 0:
                return xL
            else:
                return xM
        elif yU*yM == 0:
            if yU == 0:
                return xU
            else:
                return xM
        if i > 0:
            if xM == 0:
                eA = abs((xM-xOldM)/(xM+sys.float_info.epsilon))*100
            else:
                eA = abs((xM-xOldM)/xM)*100
            print('{:12d}\t\t{:.6f}%'.format(i+1, eA))
            if eA <= eS:
                return xM
        else:
            print('{:12d}\t\tCannot be calculated'.format(i+1))
        xOldM = xM


def drawGraph():
    x = np.array(np.arange(-10, 10, 1))
    y = f(x)
    plt.plot(x, y)
    plt.grid()
    plt.title("f(x)=x^3 - 0.18x^2 + 4.752x10^-4")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def main():
    drawGraph()
    xL = float(input("Estimate lower bound: "))
    xU = float(input("Estimate upper bound: "))
    maxIter = int(input("Enter max allowed iteration: "))
    print("Depth of the submerged ball is {:.6f}  meter".format(
        root(xL, xU, 0.5, maxIter)))
    table(xL, xU, 0.5, 20)


if __name__ == "__main__":
    main()
# x^3 - 0.18x^2 + 4.752x10^-4
