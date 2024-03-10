import numpy as np
import matplotlib.pyplot as plt

# Customer data
customer_data = np.array([
    [20, 6, 2, 386, 1],
    [16, 3, 6, 289, 1],
    [27, 6, 2, 393, 1],
    [19, 1, 2, 110, 0],
    [24, 4, 2, 280, 1],
    [22, 1, 5, 167, 0],
    [15, 4, 2, 271, 1],
    [18, 4, 2, 274, 1],
    [21, 1, 4, 148, 0],
    [16, 2, 4, 198, 0]
])

# Separate features and labels
X = customer_data[:, :-1]
y = customer_data[:, -1]

# Matrix pseudo-inverse method
X_pseudo_inv = np.linalg.pinv(np.insert(X, 0, 1, axis=1))
weights_pseudo_inv = np.dot(X_pseudo_inv, y)

# Perceptron learning
# Initialize weights and learning rate
W = np.random.rand(X.shape[1] + 1)
alpha = 0.01

# Activation function (Sigmoid)
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Function to calculate the output of the perceptron
def perceptron_output(inputs, weights):
    return sigmoid_function(np.dot(inputs, weights))

# Function to train the perceptron
def train_perceptron(inputs, weights, alpha, epochs):
    errors = []
    for epoch in range(epochs):
        error_sum = 0
        for i in range(len(inputs)):
            X_i = np.insert(inputs[i], 0, 1)
            y_i = y[i]
            output = perceptron_output(X_i, weights)
            error = y_i - output
            error_sum += error**2
            weights += alpha * error * output * (1 - output) * X_i
        errors.append(error_sum)
        if error_sum <= 0.002:
            break
    return weights, errors

# Train the perceptron for customer data
epochs_limit = 1000
trained_weights_perceptron, errors_perceptron = train_perceptron(X, W, alpha, epochs_limit)

# Display the results
print(f"Trained Weights using Pseudo-inverse: {weights_pseudo_inv}")
print(f"Trained Weights using Perceptron Learning: {trained_weights_perceptron}")

# Compare the weights obtained
weights_diff = np.abs(weights_pseudo_inv - trained_weights_perceptron)
print(f"Difference in weights: {weights_diff}")
