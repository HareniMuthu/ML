import pandas as pd
import numpy as np
from statistics import mean, variance
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def load_data(file_path, sheet_name):
    """Load data from an Excel file."""
    return pd.read_excel(file_path, sheet_name=sheet_name)

def segregate_matrices(data):
    """Separate features and target variable from the data."""
    features = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()
    target = data['Payment (Rs)'].to_numpy().reshape(-1, 1)
    return features, target

def vector_space_properties(features):
    """Calculate properties of the vector space."""
    dimensionality = features.shape[1]
    num_samples = features.shape[0]
    rank_features = np.linalg.matrix_rank(features)
    return dimensionality, num_samples, rank_features

def calculate_cost_per_product(features, target):
    """Calculate the cost per product using pseudo-inverse."""
    pseudo_inverse_features = np.linalg.pinv(features)
    return pseudo_inverse_features @ target

def calculate_model_vector(features, target):
    """Calculate the model vector for predicting cost."""
    pseudo_inverse_features = np.linalg.pinv(features)
    return pseudo_inverse_features @ target

def mark_customer_types(data):
    """Mark customers as 'RICH' or 'POOR' based on their payment amount."""
    data['Customer_Type'] = data['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
    return data

def train_classifier(data):
    """Train a classifier to predict customer types."""
    X = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()
    y = data['Customer_Type'].to_numpy()
    classifier = LogisticRegression()
    classifier.fit(X, y)
    return classifier

def stock_price_analysis(stock_data):
    """Analyze stock price data."""
    price_mean = mean(stock_data['Price'])
    price_variance = variance(stock_data['Price'])
    
    wednesday_prices = stock_data[stock_data['Day'] == 'Wednesday']['Price']
    if len(wednesday_prices) > 0:
        wednesday_mean = mean(wednesday_prices)
    else:
        wednesday_mean = None
    
    april_prices = stock_data[stock_data['Month'] == 'Apr']['Price']
    if len(april_prices) > 0:
        april_mean = mean(april_prices)
    else:
        april_mean = None

    loss_probability = len(stock_data[stock_data['Chg%'] < 0]) / len(stock_data)

    wednesday_profit_probability = None
    if len(wednesday_prices) > 0:
        wednesday_profit_probability = len(stock_data[(stock_data['Day'] == 'Wednesday') & (stock_data['Chg%'] > 0)]) / len(wednesday_prices)

    conditional_profit_probability = None
    if wednesday_profit_probability is not None:
        conditional_profit_probability = wednesday_profit_probability / (len(stock_data[stock_data['Day'] == 'Wednesday']) / len(stock_data))

    return price_mean, price_variance, wednesday_mean, april_mean, loss_probability, wednesday_profit_probability, conditional_profit_probability

def plot_chg_percent(stock_data):
    """Plot the change percentage of stock prices."""
    plt.scatter(stock_data['Day'], stock_data['Chg%'])
    plt.xlabel('Day of the Week')
    plt.ylabel('Chg%')
    plt.title('Chg% vs. Day of the Week')
    plt.show()

if __name__ == "__main__":
    # Load the purchase data and stock data from an Excel file
    purchase_data = load_data('C:\\Users\\Dell\\Desktop\\sem-4\\ML\\Lab Session1 Data.xlsx', 'Purchase data')
    stock_data = load_data('C:\\Users\\Dell\\Desktop\\sem-4\\ML\\Lab Session1 Data.xlsx', 'IRCTC Stock Price')

    # Segregate features and target variable for purchase data
    features, target = segregate_matrices(purchase_data)
    # Calculate properties of the vector space for features
    dimensionality, num_samples, rank_features = vector_space_properties(features)

    # Calculate the model vector for predicting cost
    X_model = calculate_model_vector(features, target)

    # Mark customers as 'RICH' or 'POOR' based on their payment amount
    marked_data = mark_customer_types(purchase_data)
    # Train a classifier to predict customer types
    classifier = train_classifier(marked_data)

    # Analyze stock price data
    price_mean, price_variance, wednesday_mean, april_mean, loss_probability, wednesday_profit_probability, conditional_profit_probability = stock_price_analysis(stock_data)
    # Plot the change percentage of stock prices
    plot_chg_percent(stock_data)

    # Print the results
    print("A1:")
    print(f"Dimensionality of the vector space: {dimensionality}")
    print(f"Number of samples: {num_samples}")
    print(f"Rank of Matrix features: {rank_features}")
    print("A2:")
    print(f"Model vector X for predicting cost:\n{X_model}")
    print("A3:")
    print(f"Classifier model trained successfully.")
    print("A4:")
    print(f"Mean Price: {price_mean}, Variance: {price_variance}")
    print(f"Mean Price on Wednesdays: {wednesday_mean}")
    print(f"Mean Price in April: {april_mean}")
    print(f"Probability of making a loss: {loss_probability}")
    print(f"Probability of making a profit on Wednesday: {wednesday_profit_probability}")
    print(f"Conditional probability of making profit, given that today is Wednesday: {conditional_profit_probability}")
