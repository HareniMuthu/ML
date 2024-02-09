from collections import Counter
import numpy as np

def euclidean_distance(point1, point2):
    # Calculate Euclidean distance between two points
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_nn_classifier(training_data, training_labels, new_point, k):
    distances = []

    # Calculate distances between new_point and all points in training_data
    for i, point in enumerate(training_data):
        distance = euclidean_distance(new_point, point)
        distances.append((i, distance))

    # Sort distances and get indices of k-nearest neighbors
    sorted_distances = sorted(distances, key=lambda x: x[1])
    k_nearest_indices = [index for index, _ in sorted_distances[:k]]

    # Get labels of k-nearest neighbors
    k_nearest_labels = [training_labels[index] for index in k_nearest_indices]

    # Count occurrences of each label in the k-nearest neighbors
    label_counts = Counter(k_nearest_labels)

    # Get the most common label (majority vote)
    predicted_label = label_counts.most_common(1)[0][0]

    return predicted_label

def main():
    filename = "/Users/Dell/Desktop/sem-4/ML/Iris.csv"
    iris_dataset = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Separate features and labels
    features = iris_dataset[:, :-1]
    labels = iris_dataset[:, -1]

    # Choose an example new data point
    new_point = np.array([5.1, 3.5, 1.4, 0.2])

    # Choose the value of k
    k_value = 3

    # Perform k-NN classification
    predicted_label = k_nn_classifier(features, labels, new_point, k_value)

    print("New data point:", new_point)
    print("Predicted label:", predicted_label)

if __name__ == "__main__":
    main()
