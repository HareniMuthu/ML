import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Loading project dataset
df = pd.read_csv(r'C:\Users\Dell\Desktop\sem-4\ML\assignments\archive (9)\Crop_recommendation.csv')


X_project = df[['temperature', 'humidity']].values
y_project = df['label'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_project, y_project, test_size=0.2, random_state=42)

# Generate 20 data points(training set data) consisting of 2 features (X & Y) whose values vary randomly between 1 & 10.
np.random.seed(0)
X = np.random.uniform(1, 10, (20, 2))
y = np.random.randint(0, 2, 20)

# Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1.
test_X, test_Y = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
test_data = np.column_stack((test_X.ravel(), test_Y.ravel()))

# Classify test points using kNN classifier (k = 3) trained on the project dataset
kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(X_train, y_train)
predicted_classes = kNN.predict(test_data)

# Plot the training data
colors_train = ['blue' if c == 0 else 'red' for c in y]
plt.scatter(X[:, 0], X[:, 1], color=colors_train)

# Plot the test data with predicted classes
colors_test = ['blue' if c == 0 else 'red' for c in predicted_classes]
plt.scatter(test_data[:, 0], test_data[:, 1], color=colors_test, alpha=0.05)
plt.show()
