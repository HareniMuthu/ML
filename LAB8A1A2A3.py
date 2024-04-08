import pandas as pd
import numpy as np

# Function for binning continuous features
def bin_data(data, column, bins=3, strategy='width'):
    if strategy == 'width':
        binned_data = pd.cut(data[column], bins=bins, labels=range(bins))
    elif strategy == 'frequency':
        binned_data = pd.qcut(data[column], q=bins, labels=range(bins), duplicates='drop')
    else:
        raise ValueError("Strategy not recognized. Use 'width' or 'frequency'.")
    return binned_data

# Helper function to calculate entropy
def calculate_entropy(column):
    counts = column.value_counts(normalize=True)
    entropy = -np.sum(counts * np.log2(counts))
    return entropy

# Function to calculate information gain for a feature
def information_gain(data, feature, target):
    total_entropy = calculate_entropy(data[target])
    weights = data[feature].value_counts(normalize=True)
    weighted_entropy = sum(weights[f] * calculate_entropy(data[data[feature] == f][target]) for f in data[feature].unique())
    info_gain = total_entropy - weighted_entropy
    return info_gain

# Function to find the best feature based on information gain
def find_best_feature(data, features, target):
    best_gain = 0
    best_feature = None
    for feature in features:
        gain = information_gain(data, feature, target)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    return best_feature

# Decision Tree Node
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Building the Decision Tree
def build_tree(data, features, target, depth=0, max_depth=10):
    if depth >= max_depth or len(features) == 0:
        leaf_value = data[target].mode()[0]
        return DecisionTreeNode(value=leaf_value)

    best_feature = find_best_feature(data, features, target)
    if best_feature is None:
        leaf_value = data[target].mode()[0]
        return DecisionTreeNode(value=leaf_value)

    unique_values = data[best_feature].unique()
    features = [f for f in features if f != best_feature]
    nodes = []

    for value in unique_values:
        subset = data[data[best_feature] == value]
        node = build_tree(subset, features, target, depth + 1, max_depth)
        nodes.append((value, node))

    node = DecisionTreeNode(feature=best_feature, threshold=unique_values, left=nodes[0][1], right=nodes[1][1] if len(nodes) > 1 else None)
    return node

# Predicting with the Decision Tree
def predict_sample(node, sample):
    while node.value is None:
        if sample[node.feature] in node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

def predict(tree, X):
    predictions = X.apply(lambda x: predict_sample(tree, x), axis=1)
    return predictions



# Load dataset and preprocess
data = pd.read_csv(r'C:\Users\Dell\Desktop\sem-4\ML\assignments\crop recommendation dataset\Crop_recommendation.csv')
continuous_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for column in continuous_columns:
    data[column] = bin_data(data, column, bins=3, strategy='width')

# Specify the target column name based on your dataset
target_column = 'label'  # Update this based on your dataset's target column

# Exclude the target column from the feature list
features = [col for col in data.columns if col != target_column]

# Find the best feature for the root node of the Decision Tree
root_feature = find_best_feature(data, features, target_column)
print(f"The best feature for the root node is: {root_feature}")

# Building and using the Decision Tree
tree = build_tree(data, features, target_column, max_depth=5)
predictions = predict(tree, data[features])

print(predictions.head())


