# pred_hold.py

import pandas as pd
from stockdata import StockData

# A predictor class which just always sends a "hold" signal.
# This is used to evaluate a baseline buy-and-hold strategy.


##############################################################################
class Hold_Predictor: 
    def __init__(self):
        self.name = "hold"
        
    def get_name(self):
        return self.name
    
    def get_signal(self, X):
        return pd.Series("hold", index=X.index)


        
