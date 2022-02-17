import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd


def f(x):
    return x**3 - 0.165*x**2 + 3.993*10**(-4)

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
        return (False, None)
    if f(xL)*f(xU) > 0:
        print("Error: Function doesn't changes sign on given bounds.")
        return (False, None)
    xM, xOldM = 0, 0
    for i in range(maxIter):
        print(xL, xU)
        xM = (xL+xU)/2
        yL = f(xL)
        yU = f(xU)
        yM = f(xM)
        if i > 0:
            if xM == 0:
                eA = abs((xM-xOldM)/(xM+sys.float_info.epsilon))*100
            else:
                eA = abs((xM-xOldM)/xM)*100
            if eA <= eS:
                return (True, xM)

        if yL*yM < 0:
            xU = xM
        elif yU*yM < 0:
            xL = xM
        elif yL*yM == 0:  # yL=0 or yM=0
            if yL == 0:
                return (True, xL)
            else:
                return (True, xM)
        elif yU*yM == 0:  # yU=0 or yM=0
            if yU == 0:
                return (True, xU)
            else:
                return (True, xM)
        xOldM = xM
    return (False, xM)


def table(lower_bound, upper_bound, eS, maxIter):
    if f(lower_bound)*f(upper_bound) > 0:
        print("Root can't be calculated")
        sys.exit()

    xL = np.zeros(maxIter)
    xU = np.zeros(maxIter)
    xM = np.zeros(maxIter)
    yL = np.zeros(maxIter)
    yU = np.zeros(maxIter)
    yM = np.zeros(maxIter)
    eA = np.zeros(maxIter)

    xL[0] = lower_bound
    xU[0] = upper_bound
    for i in range(maxIter):
        # Estimating new root
        xM[i] = (xL[i]+xU[i])/2
        # Finding the xL and xU for the next iteration
        yL[i] = f(xL[i])
        yU[i] = f(xU[i])
        yM[i] = f(xM[i])

        # Error analysis
        if i > 0:
            if xM[i] == 0:
                eA[i] = abs((xM[i]-xM[i-1])/(xM[i]+sys.float_info.epsilon))*100
            else:
                eA[i] = abs((xM[i]-xM[i-1])/xM[i])*100
            if eA[i] <= eS:
                break

        if i == maxIter-1:
            break
        if yL[i]*yM[i] < 0:
            xU[i+1] = xM[i]
            xL[i+1] = xL[i]
        elif yU[i]*yM[i] < 0:
            xU[i+1] = xU[i]
            xL[i+1] = xM[i]
        else:
            break

    data = {'xL': xL, 'xU': xU,
            'f(xL)': yL, 'f(xU)': yU, 'xM': xM, 'eA': eA, 'f(xM)': yM}
    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df)+1)
    print(df)


def drawGraph():
    x = np.arange(-10, 10, 1)
    y = f(x)
    plt.plot(x, y)
    plt.grid()
    plt.title("f(x)=x^3 - 0.18x^2 + 4.752x10^-4")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def take_input():
    xL = float(input("Estimate lower bound: "))
    xU = float(input("Estimate upper bound: "))
    eS = float(input('Enter max tolerance: '))
    maxIter = int(input("Enter max allowed iteration: "))
    table(xL, xU, eS, maxIter)


def main():
    # drawGraph()
    # take_input()
    table(0, 0.11, .00005, 20)
    # status, ans = root(xL, xU, eS, maxIter)
    # if status == False:
    #     print("No solution exists")
    # else:
    #     print("Solution: ", ans)


if __name__ == "__main__":
    main()
# x^3 - 0.18x^2 + 4.752x10^-4
