# pred_rfr.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from stockdata import StockData

# CJM: we could add a method for selecting hyperparameters, but it shouldn't
# really be necessary. But if we do, it's very important to make sure that
# it only uses the training data to determine them.



##############################################################################
class RFR_Predictor: 
    def __init__(self, name, sell_quantile=0.33, buy_quantile=0.66, params=None):
        self.name = name
        if params is None:
            params = {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 60}
        self.model = RandomForestRegressor(**params, random_state=42)
        self.sell_quantile = sell_quantile
        self.buy_quantile = buy_quantile
        
    def get_name(self):
        return self.name

    def get_quantiles(self):
        return self.sell_quantile, self.buy_quantile

    def train(self, trainX, trainY):
        # First fit the model.
        self.model.fit(trainX, trainY)
        self.featureImportances = self.model.feature_importances_
        
        # Now use the results on the training set to determine 66%, 33% quantiles
        # which will be used to produce buy/hold/sell signals.
        train_pred = self.model.predict(trainX)
        self.sell_cutoff = np.quantile(train_pred, self.sell_quantile)
        self.buy_cutoff = np.quantile(train_pred, self.buy_quantile)
    
    def apply(self, X):
        # return the raw result of applying this model to X,
        # which should be a DataFrame.
        return pd.Series(self.model.predict(X), index=X.index)
      
    def get_signal(self, X):
        # Like above, but returns 'buy', 'hold', or 'sell'
        # Whatever additional methods are needed from all models.
        y = self.model.predict(X)
        retval = []
        for i in range(len(y)):
            if y[i] <= self.sell_cutoff:
                retval.append("sell")
            elif y[i] >= self.buy_cutoff:
                retval.append("buy")
            else:
                retval.append("hold")
        return pd.Series(retval, index=X.index)

    def feature_importances(self):
        # This method need not be implemented in every predictor class.
        try:
            return self.featureImportances
        except:
            print("RFR_Predictor() : model must be trained before feature_importances is available.")
            return None

        
