# pred_rand.py

import pandas as pd
import numpy as np
from stockdata import StockData
import random

# This is a "predictor" which just chooses buy/hold/sell randomly.
# The idea is that one can run multiple simulations with this predictor
# to establish an empirical confidence interval for the behavior of
# a random predictor. In that way, we could reasonably answer a question
# of the form, "Is (a strategy using Predictor X) better than (the same strategy
# with a random predictor) with (say) 95% confidence?"


##############################################################################
class Random_Predictor: 
    def __init__(self, name, sell_quantile=1/3, buy_quantile=2/3, params=None):
        self.name = name
        self.sell_quantile = sell_quantile
        self.buy_quantile = buy_quantile

    def get_name(self):
        return self.name

    def get_quantiles(self):
        return self.sell_quantile, self.buy_quantile

      
    def train(self, trainX, trainY):
        # We might actually want to use the training data and estimate the
        # distribution of returns, just in case we decide to do something
        # more sophisticated than simple quantiles later on.
        pass
    
    
    def apply(self, X):
        # return the raw result of applying this model to X,
        # which should be a DataFrame.
        pass
      
    def get_signal(self, X):
        # Like above, but returns 'buy', 'hold', or 'sell'
        # Whatever additional methods are needed from all models.
        retval = []
        sigs = ['sell', 'hold', 'buy']
        for i in range(len(X)):
            x = random.random()
            if x < self.sell_quantile:
                retval.append('sell')
            elif x > self.buy_quantile:
                retval.append('buy')
            else:
                retval.append('hold')
        return pd.Series(retval, index=X.index)
        
