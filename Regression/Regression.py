import numpy as np
import matplotlib.pyplot as plt


def A(x, y, b):
    q = np.sum(y*np.exp(b*x))
    r = np.sum(np.exp(2*b*x))
    return q/r


def f(x, y, b):
    p = np.sum(x*y*(np.exp(b*x)))
    s = np.sum(x*(np.exp(2*b*x)))
    return p-(A(x, y, b)*s)


def root(x, y, xL, xU, eS=0.005, maxIter=20):
    if xL > xU:
        print("Error: Lower bound is greater than upper bound.")
        sys.exit()
    if f(x, y, xL)*f(x, y, xU) > 0:
        print("Error: Function doesn't changes sign on given bounds.")
        sys.exit()
    xM, xOldM = 0, 0
    for i in range(maxIter):
        xM = (xL+xU)/2
        yL = f(x, y, xL)
        yU = f(x, y, xU)
        yM = f(x, y, xM)
        if i > 0:
            if xM == 0:
                eA = abs((xM-xOldM)/(xM+sys.float_info.epsilon))*100
            else:
                eA = abs((xM-xOldM)/xM)*100
            if eA <= eS:
                return xM

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
        xOldM = xM
    return


def exponential(x, y):
    b = root(x, y, 20, 30)
    a = A(x, y, b)
    return (a, b)


def partialPivot(A, n, k):
    absMaxIdx = k
    for i in range(k, n):
        if abs(A[i][k]) > abs(A[absMaxIdx][k]):
            absMaxIdx = i
    if absMaxIdx != k:
        A[[k, absMaxIdx], :] = A[[absMaxIdx, k], :]


def printMatrix(A, n):
    print("A matrix:")
    for i in range(n):
        for j in range(n):
            print(f"{A[i][j]:.4f}", end=" ")
        print("")
    print("")
    print("B matrix:")
    for i in range(n):
        print(f"{A[i][n]:.4f}")
        # print(round(A[i][n], 4))
    print("")


def GaussianElimination(A, B, pivot=True, showall=True):
    n = len(A)
    x = np.zeros(n)
    A = np.append(A, B, axis=1)
    # Forward elimination
    for k in range(0, n-1):  # goes down -> (n-1) steps
        if pivot:
            partialPivot(A, n, k)
        # A[absMaxIdx][j] == 0
        if A[k][k] == 0:
            return [False, x]
        if showall:
            print('Step {:d}:'.format(k+1))
        for i in range(k+1, n):   # goes right -> i sub steps
            factor = A[i][k]/A[k][k]
            for j in range(n+1):
                A[i][j] = A[i][j] - factor*A[k][j]
            if showall:
                print('Sub Step {:d}:'.format(i-k))
                printMatrix(A, n)
    # Back substitution
    for i in range(n-1, -1, -1):
        if A[i][i] == 0:
            return [False, x]
        x[i] = A[i][n]
        for j in range(i+1, n):
            x[i] = x[i]-x[j]*A[i][j]
        x[i] = x[i]/A[i][i]
    return [True, x]


def polynomial(x, y, n):
    A = np.zeros((n+1, n+1))
    B = np.zeros((n+1, 1))
    for i in range(0, n+1):
        B[i] = np.sum((x**i)*y)

    for i in range(0, 2*n+1):
        tmp = np.sum(x**i)
        r = min(i, n)
        c = max(0, i-n)
        while r >= 0 and c <= n:
            A[r][c] = tmp
            r -= 1
            c += 1
    return GaussianElimination(A, B)


def linear(x, y):
    n = np.size(x)
    p = np.sum(x*x)
    q = np.sum(x)
    r = np.sum(x*y)
    s = np.sum(y)
    a_1 = (n*r-q*s) / (n*p - q*q)
    a_0 = (p*s-q*r) / (n*p - q*q)
    return (a_0, a_1)


def drawGraph(x, y_pred, s):
    plt.plot(x, y_pred)
    # plt.legend(s)


def main():
    # x = np.array([80, 40, -40, -120, -200, -280, -340], dtype=np.float64)
    # y = np.array([6.47, 6.24, 5.72, 5.09, 4.30, 3.33, 2.45], dtype=np.float64)
    # x = np.array([0, 0.01, 0.03, 0.05, 0.07, 0.09,
    #               0.11, 0.13, 0.15, 0.17, 0.19, 0.21])
    # y = np.array([1, 1.03, 1.06, 1.38, 2.09, 3.54,
    #              6.41, 12.6, 22.1, 39.05, 65.32, 99.78])
    x = np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.float64)
    y = np.array([1000, 550, 316, 180, 85, 56, 31], dtype=np.float64)
    plt.scatter(x, y)
    # Polynomial

    n = np.size(x)
    m = 5
    status, a = polynomial(x, y, m)
    y_pred = np.zeros(n, dtype=np.float64)
    print(a)
    for i in range(n):
        for j in range(m+1):
            y_pred[i] += a[j]*(x[i]**j)
    drawGraph(x, y_pred, "Polynomial")

    # Linear
    a_0, a_1 = linear(x, y)
    drawGraph(x,  a_0 + a_1*x, "Linear")

    # Linear Exponential
    a_0, a_1 = linear(x, np.log(y))
    a = np.exp(a_0)
    b = a_1
    drawGraph(x, a * np.exp(b*x), "Linear Exponential")

    # Exponential
    a, b = exponential(x, y)
    drawGraph(x, a * np.exp(b*x), "Exponential")

    # Power
    a_0, a_1 = linear(np.log(x), np.log(y))
    a = np.exp(a_0)
    b = a_1
    drawGraph(x, a * (x**b), "Power")

    # Saturation
    a_0, a_1 = linear(1/x, 1/y)
    a = 1/a_0
    b = a_1/a_0
    drawGraph(x,  (a*x) / (b+x), "Saturation")

    plt.legend(["Polynomial", "Linear", "Linear Exponential", "Exponential"
               "Power", "Saturation", "Original"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
