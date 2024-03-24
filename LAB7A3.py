import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset = pd.read_csv(r'C:\Users\Dell\Desktop\sem-4\ML\assignments\crop recommendation dataset\Crop_recommendation.csv')

# Split the dataset into features (X) and labels (y)
X = dataset.drop("label", axis=1)
y = dataset["label"]

# Encode labels as integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
svm_classifier = SVC()
dt_classifier = DecisionTreeClassifier()
rf_classifier = RandomForestClassifier()
catboost_classifier = CatBoostClassifier()
adaboost_classifier = AdaBoostClassifier()
xgboost_classifier = XGBClassifier()
naive_bayes_classifier = GaussianNB()

# Train classifiers
svm_classifier.fit(X_train, y_train)
dt_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)
catboost_classifier.fit(X_train, y_train)
adaboost_classifier.fit(X_train, y_train)
xgboost_classifier.fit(X_train, y_train)
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions on test set
svm_predictions = svm_classifier.predict(X_test)
dt_predictions = dt_classifier.predict(X_test)
rf_predictions = rf_classifier.predict(X_test)
catboost_predictions = catboost_classifier.predict(X_test)
adaboost_predictions = adaboost_classifier.predict(X_test)
xgboost_predictions = xgboost_classifier.predict(X_test)
naive_bayes_predictions = naive_bayes_classifier.predict(X_test)

# Calculate performance metrics
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1_score = f1_score(y_test, svm_predictions, average='weighted')

dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions, average='weighted')
dt_recall = recall_score(y_test, dt_predictions, average='weighted')
dt_f1_score = f1_score(y_test, dt_predictions, average='weighted')

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, average='weighted')
rf_recall = recall_score(y_test, rf_predictions, average='weighted')
rf_f1_score = f1_score(y_test, rf_predictions, average='weighted')

catboost_accuracy = accuracy_score(y_test, catboost_predictions)
catboost_precision = precision_score(y_test, catboost_predictions, average='weighted')
catboost_recall = recall_score(y_test, catboost_predictions, average='weighted')
catboost_f1_score = f1_score(y_test, catboost_predictions, average='weighted')

adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)
adaboost_precision = precision_score(y_test, adaboost_predictions, average='weighted')
adaboost_recall = recall_score(y_test, adaboost_predictions, average='weighted')
adaboost_f1_score = f1_score(y_test, adaboost_predictions, average='weighted')

xgboost_accuracy = accuracy_score(y_test, xgboost_predictions)
xgboost_precision = precision_score(y_test, xgboost_predictions, average='weighted')
xgboost_recall = recall_score(y_test, xgboost_predictions, average='weighted')
xgboost_f1_score = f1_score(y_test, xgboost_predictions, average='weighted')

naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
naive_bayes_precision = precision_score(y_test, naive_bayes_predictions, average='weighted')
naive_bayes_recall = recall_score(y_test, naive_bayes_predictions, average='weighted')
naive_bayes_f1_score = f1_score(y_test, naive_bayes_predictions, average='weighted')

# Convert labels back to original string labels for tabulation
y_test = label_encoder.inverse_transform(y_test)
svm_predictions = label_encoder.inverse_transform(svm_predictions)
dt_predictions = label_encoder.inverse_transform(dt_predictions)
rf_predictions = label_encoder.inverse_transform(rf_predictions)
catboost_predictions = label_encoder.inverse_transform(catboost_predictions)
adaboost_predictions = label_encoder.inverse_transform(adaboost_predictions)
xgboost_predictions = label_encoder.inverse_transform(xgboost_predictions)
naive_bayes_predictions = label_encoder.inverse_transform(naive_bayes_predictions)

# Create a dataframe to tabulate the results
results_df = pd.DataFrame({ 'Classifier': ['Support Vector Machine', 'Decision Tree', 'Random Forest', 'CatBoost', 'AdaBoost', 'XGBoost', 'Naive Bayes'], 'Accuracy': [svm_accuracy, dt_accuracy, rf_accuracy, catboost_accuracy, adaboost_accuracy, xgboost_accuracy, naive_bayes_accuracy], 'Precision': [svm_precision, dt_precision, rf_precision, catboost_precision, adaboost_precision, xgboost_precision, naive_bayes_precision], 'Recall': [svm_recall, dt_recall, rf_recall, catboost_recall, adaboost_recall, xgboost_recall, naive_bayes_recall], 'F1 Score': [svm_f1_score, dt_f1_score, rf_f1_score, catboost_f1_score, adaboost_f1_score, xgboost_f1_score, naive_bayes_f1_score]
})

# Display the results
print(results_df)
