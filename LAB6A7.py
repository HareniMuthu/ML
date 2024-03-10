import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    """
    Derivative of the sigmoid activation function
    """
    return sigmoid(x) * (1 - sigmoid(x))

def compute_error(target, output):
    """
    Compute the mean squared error
    """
    return np.mean(np.square(target - output))

def update_weights(inputs, target, weights, bias, learning_rate):
    """
    Update weights and bias based on error
    """
    # Forward propagation
    z = np.dot(weights, inputs) + bias
    output = sigmoid(z)
    
    # Calculate error
    error = output - target
    
    # Backward propagation
    delta = error * derivative_sigmoid(z)
    weight_delta = delta * inputs
    
    # Update weights and bias
    weights -= learning_rate * weight_delta
    bias -= learning_rate * delta
    
    return output, error, weights, bias

def train_AND_gate(inputs, targets, learning_rate, epochs, convergence_threshold):
    """
    Train an AND gate using a neural network
    """
    # Initialize weights and bias
    weights = np.array([0.5, 0.5])
    bias = -1.5
    
    for epoch in range(epochs):
        # Update weights for each training sample
        for i in range(len(inputs)):
            input_data = inputs[i]
            target = targets[i]
            output, error, weights, bias = update_weights(input_data, target, weights, bias, learning_rate)
        
        # Compute total error
        total_error = compute_error(targets, [sigmoid(np.dot(weights, input_data) + bias) for input_data in inputs])
        
        # Print error for each epoch (optional)
        print(f'Epoch: {epoch + 1}, Error: {total_error}')
        
        # Break loop if error is below convergence threshold
        if total_error <= convergence_threshold:
            print('Training complete!')
            return weights, bias
    
    print('Training complete!')
    return weights, bias

# Training data
inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
targets = np.array([0, 0, 0, 1])

learning_rate = 0.05
epochs = 1000
convergence_threshold = 0.002

# Train the AND gate
final_weights, final_bias = train_AND_gate(inputs, targets, learning_rate, epochs, convergence_threshold)

# Display the final weights and bias
print('Final Weights:', final_weights)
print('Final Bias:', final_bias)
