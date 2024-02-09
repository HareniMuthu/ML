import pandas as pd

# Load the Iris dataset into a pandas DataFrame
file_path = "/Users/Dell/Desktop/sem-4/ML/Iris.csv"
df = pd.read_csv(file_path)

# Define the function for one-hot encoding
def one_hot_encode(df, column_name):
    # Create a dictionary to store mappings of categories to numeric values
    category_mapping = {}
    
    # Get unique categories from the specified column
    categories = df[column_name].unique()
    
    # Initialize an empty DataFrame to store one-hot encoded values
    encoded_df = pd.DataFrame()
    
    # Iterate over each category
    for category in categories:
        # Create a new column for each category
        encoded_df[column_name + '_' + category] = (df[column_name] == category).astype(int)
        
        # Store the mapping of category to numeric value
        category_mapping[category] = len(category_mapping)
    
    return encoded_df, category_mapping

# Call the function with the loaded DataFrame and the name of the categorical column
encoded_df, category_mapping = one_hot_encode(df, 'Species')

# Display the encoded DataFrame and category mapping
print("Encoded DataFrame:")
print(encoded_df.head())

print("\nCategory Mapping:")
print(category_mapping)
