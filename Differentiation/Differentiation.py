def f(x):
    return 5*(x**2)+2*x


def differentiate(x, dx):
    return (f(x+dx)-f(x))/dx


def main():
    x = 10
    # Can be both negative and positive. More near to zero, more accurate.
    dx = 2
    print(differentiate(x, dx))


if __name__ == "__main__":
    main()
