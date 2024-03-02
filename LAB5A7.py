from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Assuming your dataset is stored in a variable named 'data'
data = pd.read_csv(r'C:\Users\Dell\Desktop\sem-4\ML\assignments\archive (9)\Crop_recommendation.csv')

X_train = data.drop('label', axis=1)  # Assuming 'label' is the target variable
y_train = data['label']

# Define the parameter grid
param_grid = {'n_neighbors': range(1, 21)}  # Range of k values from 1 to 20

# Create a kNN classifier
knn = KNeighborsClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Print the best k value
print("Best k value:", grid_search.best_params_['n_neighbors'])
