import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

df = pd.read_excel(r'C:\Users\Dell\Desktop\sem-4\ML\assignments\Lab Session1 Data.xlsx', sheet_name='IRCTC Stock Price')
# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Volume'] = df['Volume'].str.replace('K', '000').str.replace('M', '000000').str.replace('.', '').astype(float) # Convert volume to numeric

# Encode categorical variables
df['Day'] = df['Day'].astype('category')
df['Day'] = df['Day'].cat.codes

# Split dataset into features (X) and target variable (y)
X = df[['Month', 'Day', 'Open', 'High', 'Low', 'Volume']]
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the kNN model
k = 3  # Set the value of k
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Make predictions
y_pred = knn_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("R-squared (R2):", r2)
