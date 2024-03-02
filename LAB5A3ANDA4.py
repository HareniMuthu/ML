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
plt.figure(figsize=(10, 8))
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

# Train kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(np.column_stack((X_train, Y_train)), class_labels_train)

# Classify test points
predicted_classes = knn_classifier.predict(test_points)

# Scatter plot of test data with predicted class colors
plt.figure(figsize=(10, 8))
plt.scatter(test_points[predicted_classes == 0, 0], test_points[predicted_classes == 0, 1], color='blue', alpha=1, label='Predicted Class 0 (Blue)')
plt.scatter(test_points[predicted_classes == 1, 0], test_points[predicted_classes == 1, 1], color='red', alpha=1, label='Predicted Class 1 (Red)')
plt.title('Test Data Classified with kNN (k=3)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
