import pandas as pd
import numpy as np
import yfinance as yf
import itertools

def optimize_parameters(symbol1, symbol2, start_date, end_date):
    data1 = yf.download(symbol1, start=start_date, end=end_date)
    data2 = yf.download(symbol2, start=start_date, end=end_date)
    returns1 = np.log(data1['Adj Close']).diff()
    returns2 = np.log(data2['Adj Close']).diff()
    spread = returns1 - returns2

    window_sizes = list(range(5, 200, 5))
    zscore_thresholds = [0.5, 1, 1.5, 2, 2.5, 3]

    best_params = None
    best_sharpe_ratio = float('-inf')

    for window_size, zscore_threshold in itertools.product(window_sizes, zscore_thresholds):
        zscore = (spread - spread.rolling(window=window_size).mean()) / spread.rolling(window=window_size).std()

        signals = pd.Series(0, index=zscore.index)
        signals[zscore > zscore_threshold] = 1
        signals[zscore < -zscore_threshold] = -1
        signals[(zscore > -zscore_threshold) & (zscore < zscore_threshold)] = 0

        positions1, positions2 = signals.copy(), -signals.copy()
        positions = pd.concat([positions1, positions2], axis=1)
        positions.columns = [symbol1, symbol2]
        portfolio_returns = (positions.shift(1) * (returns1 + returns2)).sum(axis=1)

        annualized_return = ((1 + portfolio_returns.mean()) ** 252 - 1) * 100
        volatility = np.std(portfolio_returns, ddof=1) * np.sqrt(252) * 100
        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe_ratio = annualized_return / volatility

        if not np.isnan(sharpe_ratio) and sharpe_ratio > best_sharpe_ratio:
            best_params = (window_size, zscore_threshold)
            best_sharpe_ratio = sharpe_ratio

    return best_params