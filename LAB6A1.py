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

# Function for Step activation
def step_function(x):
    return 1 if x >= 0 else 0

# Function to calculate the output of the perceptron
def perceptron_output(inputs, weights):
    return step_function(np.dot(inputs, weights[1:]) + weights[0])  # Add a bias term

# Function to train the perceptron
def train_perceptron(inputs, weights, alpha, epochs):
    errors = []
    for epoch in range(epochs):
        error_sum = 0
        for row in inputs:
            X = row[:-1]
            y = row[-1]
            output = perceptron_output(X, weights)
            error = y - output
            error_sum += error**2
            weights[1:] += alpha * error * X
            weights[0] += alpha * error  # Update bias term
        errors.append(error_sum)
        if error_sum <= 0.002:
            break
    return weights, errors

# Train the perceptron for AND gate
epochs_limit = 1000
trained_weights, errors = train_perceptron(data_and, W, alpha, epochs_limit)

# Plot the errors
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Training Error over Epochs')
plt.show()

# Display the results
print(f"Trained Weights: {trained_weights}")
print(f"Number of epochs needed for convergence: {len(errors)}")
