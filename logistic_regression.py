import numpy as np

learning_rate = 0.01
tolerance = 0.0001

X = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([3.0, 5.0, 7.0, 9.0])
z = np.array([-2.0, 0.0, 2.0])

def forwardPass(w, b, X):
    return w*X + b

def MSE(y_pred, y_true):
    return np.mean((y_true - y_pred)**2)

def backwardPass(y_pred, y_true, X):
    pred_err = y_pred - y_true

    dw = np.mean(2*(pred_err)*X)
    db = np.mean(2*pred_err)

    print("Prediction error: ", pred_err)
    print("DB: ", db)
    print("DW: ", dw)

    return dw, db

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
#    w = 0.0
#    b = 0.0
    print(sigmoid(z))
    

    """for epoch in range(10000):
        y_pred = forwardPass(w, b, X)

        dw, db = backwardPass(y_pred, y, X)

        if abs(dw) < tolerance and abs(db) < tolerance:
            print(f"Converged at epoch {epoch}")
            break

        w = w - learning_rate * dw
        b = b - learning_rate * db

        y_pred_new = forwardPass(w,b,X)
        print(f"MSE for {epoch}: ", MSE(y, y_pred_new))

    print("Final weights: ", w)
    print("Final bias: ", b)
    """

if __name__ == "__main__":
    main()

