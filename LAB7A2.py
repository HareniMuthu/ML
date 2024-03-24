import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.exceptions import ConvergenceWarning
import warnings

# Load the dataset
data = pd.read_csv(r'C:\Users\Dell\Desktop\sem-4\ML\assignments\crop recommendation dataset\Crop_recommendation.csv')

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['label'])

# Separate features and target variable
X = data.drop('label', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for MLPRegressor
param_grid_mlp = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': uniform(loc=0, scale=0.0001),
    'learning_rate': ['constant','adaptive'],
}

# Define the parameter grid for Perceptron
param_grid_perceptron = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': uniform(loc=0, scale=0.0001),
    'max_iter': [100, 500, 1000],
    'tol': [1e-3, 1e-4, 1e-5],
}

# Suppress the convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Perform RandomizedSearchCV for MLPRegressor with increased max_iter
mlp_random_search = RandomizedSearchCV(
    MLPRegressor(random_state=42, max_iter=1000),  # Increase max_iter
    param_distributions=param_grid_mlp,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
mlp_random_search.fit(X_train_scaled, y_train)

print("Best MLPRegressor parameters found:")
print(mlp_random_search.best_params_)
print("Best MLPRegressor score found:")
print(mlp_random_search.best_score_)

# Perform RandomizedSearchCV for Perceptron
perceptron_random_search = RandomizedSearchCV(
    Perceptron(random_state=42),
    param_distributions=param_grid_perceptron,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
perceptron_random_search.fit(X_train_scaled, y_train)

print("Best Perceptron parameters found:")
print(perceptron_random_search.best_params_)
print("Best Perceptron score found:")
print(perceptron_random_search.best_score_)
