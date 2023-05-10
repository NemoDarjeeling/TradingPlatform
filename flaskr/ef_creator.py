import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
# from pypfopt import plotting

# stocks = ['NUS', 'HCSG', 'BEN', 'WST', 'SEIC', 'PB', 'FDS', 'APD', 'SPGI', 'JNJ', 'MSM', 'LANC', 'TROW', 'GGG', 'AOS']
stocks = ['AAPL','HCSG']

min_weight = 0.04
max_weight = 0.9

df = yf.download(stocks, start='2023-03-25', end='2023-04-12')

df = np.log(1+df['Adj Close'].pct_change())

def portfolio_return(weights):
  return (1+(np.dot(df.mean(),weights)))**253-1

# Scalable way to calculate Portfolio Standard Deviation
def portfolio_standard_deviation(weights):
  return np.dot(np.dot(df.cov(), weights), weights)**(1/2)*np.sqrt(250)

def weights_creator(df):
  rand = np.random.uniform(low=min_weight, high=max_weight, size=len(df.columns))
  rand /= rand.sum()
  return rand

returns = []
standard_devs = []
w = []
sharpe = []
rf = 0.0178

for i in range(100):
  weights = weights_creator(df)
  returns.append(portfolio_return(weights))
  standard_devs.append(portfolio_standard_deviation(weights))
  w.append(weights)
  sharpe.append((returns[i] - rf)/ standard_devs[i])

print("***********standard_devs**********")
print(standard_devs)
print("***********returns**********")
print(returns)
print("***********sharpe**********")
print(sharpe)


plt.figure(figsize=(10, 7))
plt.scatter(standard_devs, returns, c=sharpe, cmap='YlGnBu', marker='o', s=10, alpha=0.3)
plt.scatter(min(standard_devs), returns[standard_devs.index(min(standard_devs))], c='pink', s=500, marker='*', label='Min Vol Portfolio')
plt.scatter(standard_devs[sharpe.index(max(sharpe))], returns[sharpe.index(max(sharpe))], c='red', s=500, marker='*', label='Maximum Sharpe ratio')
plt.title('Efficient Frontier')
plt.xlabel('Standard Deviation')
plt.ylabel('Annualized Return')
plt.legend(labelspacing=1)
plt.show()

# Minimum Variance Portfolio
print('Min Standard Deviation: ', min(standard_devs))
print('Return of Min StDev Portfolio: ', returns[standard_devs.index(min(standard_devs))])
print('Weights for Min StDev Portfolio: ', w[standard_devs.index(min(standard_devs))])
print('')
# Highest Sharpe Portfolio
print('Max Sharpe: ', max(sharpe))
print('Return of Max Sharpe: ', returns[sharpe.index(max(sharpe))])
print('Weights for Max Sharpe: ', w[sharpe.index(max(sharpe))])

# plotting.plot_covariance(df.corr(), plot_correlation=True, show_tickers=True)
# plt.savefig('correl_matrix.png')