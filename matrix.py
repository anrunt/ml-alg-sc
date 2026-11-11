import numpy as np

X = np.array([
    [1.0, 0.40],
    [2.0, 0.60],
    [4.0, 0.80],
    [6.0, 0.95]
])

w = np.array([0.5, 2.0])
b = -2.0

def main():
    z = w[0] * X[:, 0] + w[1] * X[:, 1] + b

    #print(z)
    print(X)
    print(X.shape)

    print("\n")

    x_t = X.T
    print(x_t)
    print(x_t.shape)


if __name__ == "__main__":
    main()
