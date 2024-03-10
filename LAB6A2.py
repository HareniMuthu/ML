import numpy as np
import matplotlib.pyplot as plt

# AND Gate data
data_and = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

# Initial weights
W = np.array([10, 0.2, -0.75])

# Learning rate
alpha = 0.05

# Activation functions
def step_function(x):
    return 1 if x >= 0 else 0

def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return max(0, x)

# Function to calculate the output of the perceptron with different activation functions
def perceptron_output(inputs, weights, activation_function):
    output = np.dot(np.insert(inputs, 0, 1), weights)  # Add a bias term
    return activation_function(output)

# Function to train the perceptron with different activation functions
def train_perceptron(inputs, weights, alpha, epochs, activation_function):
    errors = []
    for epoch in range(epochs):
        error_sum = 0
        for row in inputs:
            X = row[:-1]
            y = row[-1]
            output = perceptron_output(X, weights, activation_function)
            error = y - output
            error_sum += error**2
            weights += alpha * error * np.insert(X, 0, 1)
        errors.append(error_sum)
        if error_sum <= 0.002:
            break
    return weights, errors

# Train the perceptron with different activation functions
epochs_limit = 1000

# Bi-Polar Step function
trained_weights_bipolar, errors_bipolar = train_perceptron(data_and, W, alpha, epochs_limit, bipolar_step_function)

# Sigmoid function
trained_weights_sigmoid, errors_sigmoid = train_perceptron(data_and, W, alpha, epochs_limit, sigmoid_function)

# ReLU function
trained_weights_relu, errors_relu = train_perceptron(data_and, W, alpha, epochs_limit, relu_function)

# Plot the errors for different activation functions
plt.plot(range(1, len(errors_bipolar) + 1), errors_bipolar, label='Bi-Polar Step', marker='o')
plt.plot(range(1, len(errors_sigmoid) + 1), errors_sigmoid, label='Sigmoid', marker='o')
plt.plot(range(1, len(errors_relu) + 1), errors_relu, label='ReLU', marker='o')

plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Training Error over Epochs (Different Activation Functions)')
plt.legend()
plt.show()
