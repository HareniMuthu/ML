import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load data from CSV (replace 'your_data.csv' with your actual file path)
data = pd.read_csv(r"C:\Users\Dell\Desktop\sem-4\ML\assignments\archive (9)\Crop_recommendation.csv")

# Separate features and target variable
X = data.drop(columns=['label', 'K'])  # Assuming 'class' and 'ImageName' are the target and irrelevant columns
y = data['label']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define and train the MLP model
mlp = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(10, 5), random_state=42)
mlp.fit(X_train, y_train)

# Make predictions on test data
y_pred = mlp.predict(X_test)

# Evaluate model performance
print(classification_report(y_test, y_pred))
