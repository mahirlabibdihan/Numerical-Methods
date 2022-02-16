import sys


def f(x):
    return (1 / x)


def trapezoidal(a, b, eS, maxIter):
    n = 1
    oldS = 0
    for n in range(1, maxIter):
        h = (b - a) / n
        s = (f(a) + f(b))
        for i in range(1, n):
            s += 2 * f(a + i * h)
        s *= h / 2
        if n > 1:
            if s == 0:
                eA = abs((s-oldS)/(s+sys.float_info.epsilon))*100
            else:
                eA = abs((s-oldS)/s)*100
            if eA < eS:
                return s
        oldS = s
    return oldS


def main():
    x0 = int(input())
    xn = int(input())
    print("Value of integral is ", "%.4f" % trapezoidal(x0, xn, .0005, 20))


if __name__ == "__main__":
    main()
