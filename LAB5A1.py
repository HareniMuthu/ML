import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Read the dataset
dataset = pd.read_csv(r'C:\Users\Dell\Desktop\sem-4\ML\assignments\archive (9)\Crop_recommendation.csv') 

# Assuming 'label' is the column containing class labels and other columns are features
X = dataset.drop('label', axis=1).values
y = dataset['label'].values

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN classifier
neighbour = KNeighborsClassifier(n_neighbors=3)
neighbour.fit(X_train, y_train)

# Test accuracy of kNN classifier
accuracy = neighbour.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict using kNN classifier
predictions = neighbour.predict(X_test)
#print("Predictions:", predictions)

# Evaluate confusion matrix and performance metrics
conf_matrix = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=neighbour.classes_)
disp.plot(cmap=plt.cm.Blues, values_format=".0f")
plt.title('Confusion Matrix - Test Set')
plt.show()

precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print("Confusion Matrix:")
print(conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
