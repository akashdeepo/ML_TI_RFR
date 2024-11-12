# simulate_trading.py

from pred_rand import Random_Predictor
from stockdata import StockData
from pred_rfr import RFR_Predictor
import numpy as np
import pandas as pd
import os

# Default trading parameters
DEFAULT_TURNOVER_CONSTRAINT = 0.004
DEFAULT_RANDOM_SIMS = 1000
DEFAULT_START_VALUE = 10000

###############################################################################
def simulate_trading(SD, Model, turnover_constraint=DEFAULT_TURNOVER_CONSTRAINT, start_value=DEFAULT_START_VALUE):
    """
    Simulate trading with improved tracking and validation.
   
    Parameters:
    SD : StockData object
    Model : Predictor model
    turnover_constraint : float, maximum position change per period
    start_value : float, initial portfolio value
   
    Returns:
    portfolio : DataFrame with trading history and returns
    """
    testX, testY = SD.get_test_set()
    signals = Model.get_signal(testX)

    # Validate test set dates
    date_ranges = SD.get_train_test_dates()
    print(f"Running simulation from {date_ranges['test_start']} to {date_ranges['test_end']}")

    # Initialize portfolio DataFrame
    portfolio = pd.DataFrame(
        index=testX.index,
        columns=['signal', 'value', 'cash', 'shares_held', 'shares_traded', 'price']
    )

    # Initialize positions
    base_shares = start_value / SD.price_at_time(testX.index[0])
    cash = 0
    shares = base_shares

    for t in testX.index:
        current_price = SD.price_at_time(t)
        portfolio_value = shares * current_price + cash
        max_trade_value = portfolio_value * turnover_constraint

        # Record price and current signal
        portfolio.loc[t, 'price'] = current_price
        portfolio.loc[t, 'signal'] = signals[t]

        if signals[t] == "sell" and shares > 0:
            shares_to_sell = min(shares, max_trade_value / current_price)
            cash += shares_to_sell * current_price
            portfolio.loc[t, 'shares_traded'] = -shares_to_sell
            shares -= shares_to_sell
        elif signals[t] == "buy" and cash > 0:
            shares_to_buy = min(max_trade_value / current_price, cash / current_price)
            shares += shares_to_buy
            portfolio.loc[t, 'shares_traded'] = shares_to_buy
            cash -= shares_to_buy * current_price
        else:
            portfolio.loc[t, 'shares_traded'] = 0.0

        # Update portfolio metrics
        portfolio.loc[t, 'value'] = cash + shares * current_price
        portfolio.loc[t, 'cash'] = cash
        portfolio.loc[t, 'shares_held'] = shares

    # Ensure timestamp alignment
    portfolio['returns'] = portfolio['value'].pct_change()
    portfolio.dropna(subset=['returns'], inplace=True)  # Drop initial NaN values for returns

    return portfolio

###############################################################################
def simulate_baseline(SD, turnover_constraint=DEFAULT_TURNOVER_CONSTRAINT, start_value=DEFAULT_START_VALUE):
    """
    Simulate buy-and-hold strategy.
    
    Parameters:
    -----------
    SD : StockData object
    turnover_constraint : float, maximum position change per period
    start_value : float, initial portfolio value
    
    Returns:
    --------
    portfolio : pd.DataFrame
        DataFrame containing baseline portfolio values
    """
    testX, _ = SD.get_test_set()
    portfolio = pd.DataFrame(index=testX.index)
    initial_shares = start_value / SD.price_at_time(testX.index[0])

    # Calculate portfolio values at each point based on initial shares held
    prices = pd.Series(index=testX.index)
    for t in testX.index:
        prices[t] = SD.price_at_time(t)
    
    portfolio['value'] = initial_shares * prices
    portfolio['shares_held'] = initial_shares
    portfolio['returns'] = portfolio['value'].pct_change()
    
    return portfolio
###############################################################################
def simulate_random_trading(SD, alpha=0.05, num_sims=DEFAULT_RANDOM_SIMS, turnover_constraint=DEFAULT_TURNOVER_CONSTRAINT):
    """
    Simulate multiple random trading strategies for confidence intervals.
    """
    RandModel = Random_Predictor("rand", sell_quantile=1/3, buy_quantile=2/3)
    rand_results = []

    for i in range(num_sims):
        rp = simulate_trading(SD, RandModel, turnover_constraint=turnover_constraint)
        rand_results.append(rp['value'].copy())

    # Convert to numpy array for quantile calculation
    RR = np.array(rand_results)
    upper = np.quantile(RR, 1.0 - alpha, axis=0)
    lower = np.quantile(RR, alpha, axis=0)
    Index = rp.index

    Upper = pd.DataFrame(upper, index=Index, columns=['value'])
    Lower = pd.DataFrame(lower, index=Index, columns=['value'])
    return Upper, Lower
