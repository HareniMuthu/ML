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

# Activation function (Step)
def step_function(x):
    return 1 if x >= 0 else 0

# Function to calculate the output of the perceptron
def perceptron_output(inputs, weights):
    return step_function(np.dot(np.insert(inputs, 0, 1), weights))  # Add a bias term

# Function to train the perceptron with varying learning rates
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
    return errors

# Varying learning rates
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Plot the number of iterations against learning rates
plt.figure(figsize=(10, 6))
for alpha in learning_rates:
    errors = train_perceptron(data_and, W, alpha, 1000)
    plt.plot(range(1, len(errors) + 1), errors, label=f'Learning Rate = {alpha}', marker='o')

plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Training Error over Epochs for Different Learning Rates')
plt.legend()
plt.show()
