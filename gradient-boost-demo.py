import numpy as np

X = np.array([1., 2., 3., 4.])
y = np.array([3., 5., 7., 9.])

learning_rate = 0.5

def forwardPass(y):
    return np.mean(y)

def MSE(y_pred, y_true): 
    return np.mean((y_pred - y_true)**2)

def residuals_func(y_true, y_pred):
    return y_true - y_pred

def tree(X, residuals):
    mean_left = residuals[X <= 2.5].mean()
    mean_right = residuals[X > 2.5].mean()

    return np.where(X <= 2.5, mean_left, mean_right)

def main():
    y_pred = forwardPass(y)
    print("First predictions: ", y_pred)

    print("First MSE: ", MSE(y_pred, y))

    residuals = residuals_func(y, y_pred)
    print("Residuals: ", residuals)

    new_predictions = y_pred + learning_rate * tree(X, residuals)
    print("New predictions: ", new_predictions)

    new_mse = MSE(new_predictions, y)
    print("New MSE: ", new_mse)

if __name__ == "__main__":
    main()

