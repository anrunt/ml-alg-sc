import numpy as np

learning_rate = 0.01
tolerance = 0.0001

X = np.array([
    [1.0, 0.40],
    [2.0, 0.60],
    [4.0, 0.80],
    [6.0, 0.95]
])

y = np.array([0., 0., 1., 1.])

def forwardPass(w, b, X):
    z = X @ w + b
    return sigmoid(z)

# y = y_true
def BCE(y, p):
    return np.mean(-(y*np.log(p) + (1-y)*np.log(1-p)))

def backwardPass(p,y,X):
    pred_err = p - y

    dw = X.T @ (pred_err) / len(y)
    db = np.mean(pred_err)

    print("Prediction error: ", pred_err)
    print("DB: ", db)
    print("DW: ", dw)

    return dw, db

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    w = np.zeros(X.shape[1])
    b = 0.0

    for epoch in range(10000):
        p_pred = forwardPass(w, b, X)

        dw, db = backwardPass(p_pred, y, X)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        p_new = forwardPass(w,b,X)
        print(f"BCE for {epoch}: ", BCE(y, p_new))

    print("Final weights: ", w)
    print("Final bias: ", b)

if __name__ == "__main__":
    main()
