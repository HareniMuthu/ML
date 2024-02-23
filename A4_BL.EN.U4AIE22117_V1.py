import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Read the dataset
dataset = pd.read_csv(r'C:\Users\Dell\Desktop\sem-4\ML\archive (9)\Crop_recommendation.csv')  

# Assuming 'label' is the column containing class labels and other columns are features
X = dataset.drop('label', axis=1).values
y = dataset['label'].values

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to calculate mean for each class
def calculate_class_mean(features, labels, target_class):
    class_indices = np.where(labels == target_class)
    class_features = features[class_indices]
    return np.mean(class_features, axis=0)

# Function to calculate spread for each class
def calculate_class_spread(features, labels, target_class):
    class_indices = np.where(labels == target_class)
    class_features = features[class_indices]
    return np.std(class_features, axis=0)

# Function to calculate distance between mean vectors of two classes
def calculate_interclass_distance(mean_vector1, mean_vector2):
    return np.linalg.norm(mean_vector1 - mean_vector2)

# Assuming the expanded list of classes
classes = ['coffee', 'jute', 'cotton', 'coconut', 'papaya', 'orange', 'apple', 'muskmelon', 'watermelon', 
           'grapes', 'mango', 'banana', 'pomegranate', 'lentil', 'blackgram', 'mungbean', 'mothbeans', 
           'pigeonpeas', 'kidneybeans', 'chickpea', 'maize', 'rice']

class_means = {}
class_spreads = {}

for target_class in classes:
    class_means[target_class] = calculate_class_mean(X_train, y_train, target_class)
    class_spreads[target_class] = calculate_class_spread(X_train, y_train, target_class)

interclass_distances = {}

for i, class1 in enumerate(classes):
    for j, class2 in enumerate(classes):
        if i < j:
            interclass_distances[(class1, class2)] = calculate_interclass_distance(class_means[class1], class_means[class2])

print("Interclass distances:")
for classes, distance in interclass_distances.items():
    print(f"Distance between {classes[0]} and {classes[1]}: {distance}")

# Plot histogram for each feature
for feature_name in dataset.columns[:-1]:  # Exclude the last column (label)
    feature_values = dataset[feature_name].values
    plt.hist(feature_values, bins=10)  # Adjust bins as needed
    plt.xlabel(f'Feature {feature_name}')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Feature {feature_name}')
    plt.show()

    mean_feature = np.mean(feature_values)
    variance_feature = np.var(feature_values)

    print(f"Mean of Feature {feature_name}: {mean_feature}")
    print(f"Variance of Feature {feature_name}: {variance_feature}")

# Plot Minkowski distance for two feature vectors
# Assuming feature_vectors1 and feature_vectors2 are two feature vectors
feature_vectors1 = X_train[0]  # Assuming X_train is defined
feature_vectors2 = X_train[1]  # Assuming X_train has at least two samples

r_values = range(1, 11)
distances = []

for r in r_values:
    distance = minkowski(feature_vectors1, feature_vectors2, p=r)
    distances.append(distance)

plt.plot(r_values, distances)
plt.xlabel('r values')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. r values')
plt.show()

# Train kNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Test accuracy of kNN classifier
accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict using kNN classifier
predictions = neigh.predict(X_test)
print("Predictions:", predictions)

# Compare kNN (k=1) and kNN (k=3)
accuracy_scores = []
k_values = range(1, 12)

for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    accuracy = neigh.score(X_test, y_test)
    accuracy_scores.append(accuracy)

plt.plot(k_values, accuracy_scores)
plt.xlabel('k values')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k values')
plt.show()

# Evaluate confusion matrix and performance metrics
conf_matrix = confusion_matrix(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print("Confusion Matrix:")
print(conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
