import numpy as np

learning_rate = 0.01
tolerance = 0.0001

X = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([3.0, 5.0, 7.0, 9.0])

def forwardPass(w, b, X):
    return w*X + b

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def backwardPass(X, y_true, y_pred):
    pred_err = y_pred - y_true

    dw = 2 * np.mean(pred_err * X)
    db = 2 * np.mean(pred_err)

    print("Prediction error: ", pred_err)
    print("DB: ", db)
    print("DW: ", dw)

    return dw, db

def main():
    w = 0.0
    b = 0.0

    for epoch in range(10000):
        y_pred = forwardPass(w,b, X)

        dw, db = backwardPass(X, y, y_pred)

        if abs(dw) < tolerance and abs(db) < tolerance:
            break

        w = w - learning_rate * dw
        b = b - learning_rate * db

        y_pred_new = forwardPass(w,b,X)
        print(f"MSE for {epoch}: ", MSE(y, y_pred_new))


    print("Final weights: ", w)
    print("Final bias: ", b)

if __name__ == "__main__":
    main()
