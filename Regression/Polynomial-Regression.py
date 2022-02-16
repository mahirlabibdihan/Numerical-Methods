import numpy as np
import matplotlib.pyplot as plt


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


def drawGraph(x, y, y_pred):
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()


def main():
    # x = np.array([80, 40, -40, -120, -200, -280, -340], dtype=np.float64)
    # y = np.array([6.47, 6.24, 5.72, 5.09, 4.30, 3.33, 2.45], dtype=np.float64)
    x = np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.float64)
    y = np.array([1000, 550, 316, 180, 85, 56, 31], dtype=np.float64)
    m = 2
    status, a = polynomial(x, y, m)
    n = np.size(x)

    y_pred = np.zeros(n, dtype=np.float64)
    print(a)
    for i in range(n):
        for j in range(m+1):
            y_pred[i] += a[j]*(x[i]**j)
    drawGraph(x, y, y_pred)


if __name__ == "__main__":
    main()
