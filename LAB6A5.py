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

# Initialize weights and learning rate
W = np.random.rand(customer_data.shape[1] - 1)
alpha = 0.01

# Activation function (Sigmoid)
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Function to calculate the output of the perceptron
def perceptron_output(inputs, weights):
    return sigmoid_function(np.dot(np.insert(inputs, 0, 1), weights))  # Add a bias term

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
            weights += alpha * error * output * (1 - output) * np.insert(X, 0, 1)
        errors.append(error_sum)
        if error_sum <= 0.002:
            break
    return weights, errors

# Train the perceptron for customer data
epochs_limit = 1000
trained_weights_customer, errors_customer = train_perceptron(customer_data[:, :-1], W, alpha, epochs_limit)

# Plot the errors
plt.plot(range(1, len(errors_customer) + 1), errors_customer, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Training Error over Epochs for Customer Data')
plt.show()

# Display the results
print(f"Trained Weights for Customer Data: {trained_weights_customer}")
print(f"Number of epochs needed for convergence: {len(errors_customer)}")
