learning_rate = 0.1  # our learning rate


def loss_function(w):
    return (w - 3) ** 2


def derivative_loss_function(w):
    return 2 * (w - 3)


def main():
    w = 0.0  # Starting weigh
    for epoch in range(20):
        der = derivative_loss_function(w)

        w_new = w - learning_rate * der

        w = w_new

        current_loss = loss_function(w)

        print(f"Epoch {epoch + 1}: w = {w:.4f}, loss = {current_loss:.4f}")


if __name__ == "__main__":
    main()
