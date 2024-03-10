import numpy as np
import matplotlib.pyplot as plt

# Function to define perceptron model
def perceptron(inputs, weights, activation):
    inputs = np.atleast_2d(inputs)  # Ensure inputs is a 2D array for dot product
    z = np.dot(weights, inputs.T)  # Calculate weighted sum (including bias)

    # Apply chosen activation function
    if activation == "step":
        output = 1 if z > 0 else 0
    elif activation == "sigmoid":
        output = 1 / (1 + np.exp(-z))
    elif activation == "relu":
        output = max(0, z)
    elif activation == "bipolar_step":
        output = 1 if z > 0 else -1
    else:
        raise ValueError("Invalid activation function provided.")

    return output

# Function to train perceptron model
def train_perceptron(data, target, epochs, learning_rate, initial_weights, activation):
    weights = initial_weights[:len(data[0])]  # Take relevant weights based on input dimension
    errors = []

    for epoch in range(epochs):
        total_error = 0

        for x, y in zip(data, target):
            predicted = perceptron(x, weights, activation)
            error = y - predicted
            total_error += error**2

            # Update weights based on error
            weights += learning_rate * error * x

        # Calculate average error for the epoch
        average_error = total_error / len(data)
        errors.append(average_error)

        # Check for convergence
        if average_error <= 0.002:
            print(f"Converged in {epoch+1} epochs!")
            break

    return weights, 0.0, errors  # Assuming bias is zero

# Function to plot errors vs epochs
def plot_errors(epochs, errors, title):
    plt.plot(epochs, errors)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Constants
W0 = 10
W1 = 0.2
W2 = -0.75
bias = 0  # Assuming bias is zero
learning_rate = 0.05

# Sample training data and target (replace with actual data)
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([0, 0, 0, 1])

# Activation functions to test
activations = ["step", "sigmoid", "relu", "bipolar_step"]
for activation in activations:
    weights, bias, errors = train_perceptron(data, target, 1000, learning_rate, np.array([W0, W1, W2]), activation)
    print(f"Activation Function: {activation}")
    print(f"Final Weights: {weights}")
    print(f"Final Bias: {bias}")

    # Plot errors vs epochs
    plot_errors(range(1, len(errors)+1), errors, f"Perceptron Error ({activation})")
