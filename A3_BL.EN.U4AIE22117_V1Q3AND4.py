import pandas as pd
import statistics

# Load data from Excel file
df = pd.read_excel(r'C:\Users\Dell\Desktop\sem-4\ML\Lab Session1 Data.xlsx', sheet_name='IRCTC Stock Price')

# Calculate the mean and variance of the Price data present in column D
price_mean = df['Price'].mean()
price_variance = df['Price'].var()
print("Mean of Price data:", price_mean)
print("Variance of Price data:", price_variance)

# Select the price data for all Wednesdays and calculate the sample mean
wednesday_prices = df[df['Day'] == 'Wed']['Price']
if not wednesday_prices.empty:
    wednesday_mean = wednesday_prices.mean()
    print("\nSample mean of Wednesday price data:", wednesday_mean)
    print("Comparison with population mean:", "Greater" if wednesday_mean > price_mean else "Equal/Lesser")
else:
    print("\nNo data available for Wednesdays.")

# Select the price data for the month of April and calculate the sample mean
april_prices = df[df['Month'] == 'Apr']['Price']
if not april_prices.empty:
    april_mean = april_prices.mean()
    print("\nSample mean of April price data:", april_mean)
    print("Comparison with population mean:", "Greater" if april_mean > price_mean else "Equal/Lesser")
else:
    print("\nNo data available for the month of April.")

# Probability of making a loss over the stock (negative Chg%)
loss_probability = df['Chg%'].apply(lambda x: 1 if x < 0 else 0).mean()
print("\nProbability of making a loss over the stock:", loss_probability)

# Probability of making a profit on Wednesday
if not wednesday_prices.empty:
    wednesday_profit_probability = df[(df['Day'] == 'Wed') & (df['Chg%'] > 0)].shape[0] / wednesday_prices.shape[0]
    print("Probability of making a profit on Wednesday:", wednesday_profit_probability)
else:
    print("No data available for Wednesdays.")

# Conditional probability of making profit, given that today is Wednesday
if not wednesday_prices.empty:
    conditional_profit_probability = df[(df['Day'] == 'Wed') & (df['Chg%'] > 0)].shape[0] / wednesday_prices.shape[0]
    print("Conditional probability of making profit, given that today is Wednesday:", conditional_profit_probability)
else:
    print("No data available for Wednesdays.")

# Make a scatter plot of Chg% data against the day of the week
import matplotlib.pyplot as plt
plt.scatter(df['Day'], df['Chg%'])
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Scatter plot of Chg% data against the day of the week')
plt.xticks(rotation=45)
plt.show()
