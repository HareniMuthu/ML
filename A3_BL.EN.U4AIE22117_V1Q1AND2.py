import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def classifier(df):
    features = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    X = df[features]
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    df['Predicted Category'] = classifier.predict(X)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    print("Accuracy of the classifier:", round(accuracy * 100, 2), "%")
    return df

# Load Excel file into a pandas DataFrame
df = pd.read_excel(r'C:\Users\Dell\Desktop\sem-4\ML\Lab Session1 Data.xlsx')

# Extract data into a matrix (NumPy array)
columns = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
A = df[columns].values
row, col = A.shape
print("Dimensionality of A:", col)
print("Number of vectors in A:", row)

C = df[["Payment (Rs)"]].values

rank_A = np.linalg.matrix_rank(A)
print("Rank of A:", rank_A)

pinv_A = np.linalg.pinv(A)

print("\nPseudo-inverse of A:")
print(pinv_A)
print("\n")

X = pinv_A @ C

print(f"Cost of one candy: Rs {round(X[0][0])}")
print(f"Cost of one kg mango: Rs {round(X[1][0])}")
print(f"Cost of one milk packet: Rs {round(X[2][0])}")

# Creating a new column 'Category' based on Payment amount
df['Category'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
df = classifier(df)
print("\nData with Predicted Categories:")
print(df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Category', 'Predicted Category']])
