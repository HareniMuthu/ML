import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate random training data
np.random.seed(42)
num_points = 20
X_train = np.random.randint(1, 10, size=num_points)
Y_train = np.random.randint(1, 10, size=num_points)
class_labels_train = np.random.randint(0, 2, size=num_points)

# Scatter plot of training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[class_labels_train == 0], Y_train[class_labels_train == 0], color='blue', label='Class 0 (Blue)')
plt.scatter(X_train[class_labels_train == 1], Y_train[class_labels_train == 1], color='red', label='Class 1 (Red)')
plt.title('Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# Generate test data
x_test = np.arange(0, 10.1, 0.1)
y_test = np.arange(0, 10.1, 0.1)
X_test, Y_test = np.meshgrid(x_test, y_test)
test_points = np.column_stack((X_test.ravel(), Y_test.ravel()))

# Define different values of k
k_values = [1, 3, 5, 7]

# Plot test data with class boundary lines for each value of k
plt.figure(figsize=(14, 10))
for i, k in enumerate(k_values, start=1):
    # Train kNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(np.column_stack((X_train, Y_train)), class_labels_train)

    # Classify test points
    predicted_classes = knn_classifier.predict(test_points)

    # Plot test data with predicted class colors
    plt.subplot(2, 2, i)
    plt.scatter(test_points[predicted_classes == 0, 0], test_points[predicted_classes == 0, 1], color='blue', label='Predicted Class 0 (Blue)')
    plt.scatter(test_points[predicted_classes == 1, 0], test_points[predicted_classes == 1, 1], color='red', label='Predicted Class 1 (Red)')
    plt.title(f'Test Data Classified with kNN (k={k})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
