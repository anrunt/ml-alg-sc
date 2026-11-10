import numpy as np

learning_rate = 0.01
tolerance = 0.0001

X = np.array([1., 2., 3., 4., 5., 6.])
y = np.array([0., 0., 0., 1., 1., 1.])

def forwardPass(w, b, X):
    z = w*X + b
    return sigmoid(z)

# y = y_true
def BCE(y, p):
    return np.mean(-(y*np.log(p) + (1-y)*np.log(1-p)))

def backwardPass(p,y,X):
    pred_err = p - y

    dw = np.mean((pred_err)*X)
    db = np.mean(pred_err)

    print("Prediction error: ", pred_err)
    print("DB: ", db)
    print("DW: ", dw)

    return dw, db

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    w = 0.0
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

