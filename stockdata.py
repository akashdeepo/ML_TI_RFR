# stock_data.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import copy

"""
  CJM: I originally had written this class to perform a built-in mean 0, std 1 scaling.
  I've since changed my mind - I think the model should be responsible for doing its
  own scaling, if necessary. Some models, like RFR, don't require any scaling and doing
  it here could potentially cause some confusion/bugs to creep in.
"""
    


class StockData:
    def __init__(self,
                 filepath,
                 technical_indicator_list=[],
                 test_frac=0.2,
                 tod_start='09:00',
                 tod_end='14:30',
                 delta_t=1):
        """
        Initialize StockData with improved timezone and dividend handling.
       
        Parameters:
        filepath : str, path to CSV file
        technical_indicator_list : list of technical indicators to compute
        test_frac : float, fraction of data to use for testing
        tod_start : str, trading day start time (CST)
        tod_end : str, trading day end time (CST)
        delta_t : int, prediction time delta
        """
        self.delta_t = delta_t
        self.tod_start = tod_start
        self.tod_end = tod_end
       
        if not self.load_file(filepath):
            return
           
        # List of known dividend dates and amounts
        dividends = [{'ex_date': '2024-09-20', 'amount': 1.745531}]  # Add more dates as needed

        # Adjust prices for dividends
        self.adjust_for_dividends(dividends)

        self.preprocess_data()
        self.compute_indicators(technical_indicator_list)
        self.data.dropna(inplace=True)
        self.apply_TOD_filter()
        self.train_size = int(len(self.data)*(1-test_frac))

    def adjust_for_dividends(self, dividends):
        """
        Adjusts historical prices for dividends to ensure continuity.
        
        Parameters:
        dividends : list of dicts
            List of dividends with 'ex_date' and 'amount' keys.
        """
        for dividend in dividends:
            ex_date = pd.Timestamp(dividend['ex_date'])
            amount = dividend['amount']

            # Find the last price before the ex-dividend date
            last_price_before_dividend = self.data.loc[:ex_date - pd.Timedelta(days=1), 'close'].iloc[-1]

            # Calculate the adjustment factor
            adjustment_factor = (last_price_before_dividend - amount) / last_price_before_dividend

            # Apply the adjustment factor to prices before the ex-dividend date
            self.data.loc[:ex_date - pd.Timedelta(days=1), ['open', 'high', 'low', 'close']] *= adjustment_factor

            print(f"Adjusted prices before {ex_date} by factor {adjustment_factor:.5f} for dividend of {amount}")

    def apply_TOD_filter(self):
        """Apply time-of-day filter with improved timezone handling."""
        if self.data.index.tz is None:
            print("Note: Timestamps are assumed to be in CST")
       
        self.data = self.data.between_time(self.tod_start, self.tod_end)
        if len(self.data) == 0:
            raise ValueError(f"No data found between {self.tod_start} and {self.tod_end}")

    def load_file(self, filepath):
        try:
            self.data = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
            return True
        except:
            print(f"StockData.load_file(): failed to load {filepath}.")
            self.data = None
            return False

    def preprocess_data(self):
        VOL_SCALE_WINDOW = 60
        self.data.sort_values(by='timestamp', ascending=True, inplace=True)
       
        # Convert open, high, low, close columns to be log-relative to the previous close
        for col in ['open', 'high', 'low', 'close']:
            self.data[f'lr_{col}'] = np.log(self.data[col] / self.data[col].shift(1))

        self.target_column_name = f'lr_close_(t+{self.delta_t})'
        self.data[self.target_column_name] = self.data['lr_close'].shift(-self.delta_t)

        rolling_mean = self.data['volume'].rolling(VOL_SCALE_WINDOW).mean()
        rolling_std = self.data['volume'].rolling(VOL_SCALE_WINDOW).std()
        self.data[f'volz{VOL_SCALE_WINDOW}'] = (self.data['volume'] - rolling_mean) / rolling_std
       
        self.data.dropna(inplace=True)
    
    def get_train_test_dates(self):
        """
        Retrieve the start and end dates for the train and test sets.

        Returns:
        --------
        dict : Contains train and test start and end dates
        """
        train_start = self.data.index[0]
        train_end = self.data.index[self.train_size - 1]
        test_start = self.data.index[self.train_size]
        test_end = self.data.index[-1]
       
        return {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        }

    # Additional methods for compute_indicators, get_train_set, get_test_set, etc. go here

        
                
    ########################################################################
    def compute_indicators(self, technical_indicator_list):
        ind_method_dict = {"sma":self.calculate_sma,
                           "ema":self.calculate_ema,
                           "macd":self.calculate_macd,
                           "rsi": self.calculate_rsi,
                           "boll":  self.calculate_bollinger_bands,
                           "so": self.calculate_stochastic_oscillator,
                           "fib": self.calculate_fibonacci_retracement,
                           "adx" : self.calculate_adx,
                           "obv" : self.calculate_obv,
                           "wrobv" : self.calculate_wrobv,
                           "cci" : self.calculate_cci,
                           "ichi" : self.calculate_ichimoku
                           }
        for ti in technical_indicator_list:
            if ti in ind_method_dict:
                ind_method_dict[ti]()
            else:
                print(f"WARNING: StockData.compute_indicators(): unknown indicator '{ti}'")

    ########################################################################
    def load_file(self, filepath):
        try:
            self.data = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
            return True
        except:
            print(f"StockData.load_file(): failed to load {filepath}.")
            self.data = None
            return False

    ########################################################################
    def preprocess_data(self):
        VOL_SCALE_WINDOW = 60
        self.data.sort_values(by='timestamp', ascending=True, inplace=True)
        
        # Convert open, high, low, close columns to be log-relative to the previous close.(Updated as per Dr.Rachev instructions)
        for col in ['open', 'high', 'low', 'close']:
            self.data[f'lr_{col}'] = np.log(self.data[col] / self.data[col].shift(1))

            
        # The target column is lr_close, shifted by -self.delta_t because the data are
        # sorted ascending in time; thus, for example, the 12:01 lr_close will land
        # in the 12:00 row.
        self.target_column_name = f'lr_close_(t+{self.delta_t})'
        self.data[self.target_column_name] = self.data['lr_close'].shift(-self.delta_t)

        rolling_mean = self.data['volume'].rolling(VOL_SCALE_WINDOW).mean()
        rolling_std = self.data['volume'].rolling(VOL_SCALE_WINDOW).std()
        self.data[f'volz{VOL_SCALE_WINDOW}'] = (self.data['volume'] - rolling_mean) / rolling_std
        
        self.data.dropna(inplace=True)


        
    #######################################################################
    def apply_TOD_filter(self):
        self.data = self.data.between_time(self.tod_start, self.tod_end)

    #######################################################################
    def feature_column_names(self):
        cols = []
        nonfeatures = ['open', 'high', 'low', 'close', 'volume'] + [self.target_column_name]
        for c in self.data.columns:
            if not(c in nonfeatures):
                cols.append(c)
        return cols

    #######################################################################
    def get_train_set(self):
        # Return trainX, trainY DataFrame and Series.
        features = self.feature_column_names()
        t = self.target_column_name
        trainX = self.data[features][:self.train_size].copy()
        trainY = self.data[t][:self.train_size].copy()
        return trainX, trainY
      
    #######################################################################
    def get_test_set(self):
        # Return testX, testY DataFrame and Series.
        features = self.feature_column_names()
        t = self.target_column_name
        testX = self.data[features][self.train_size:].copy()
        testY = self.data[t][self.train_size:].copy()
        return testX, testY
      
    #######################################################################
    def get_train_date_range(self):
        # Returns t0,t1 Timestamp objects.
        t0 = self.data.index[0]
        t1 = self.data.index[self.train_size-1]
        return t0,t1
      
    #######################################################################
    def get_test_date_range(self):
        # Returns t0,t1 Datetime objects.
        t0 = self.data.index[self.train_size]
        t1 = self.data.index[-1]
        return t0, t1
    
    #######################################################################
    def get_train_index_list(self):
        return self.data.index[:self.train_size]

    #######################################################################
    def get_test_index_list(self):
        return self.data.index[self.train_size:]
    
    #######################################################################
    def price_at_time(self, t): 
        # 't' being a Timestamp object, or similar.
        return self.data['close'].loc[t]
      
    #######################################################################
    def features_at_time(self, t):
        # A tuple of the features, including technical indicators
        features = self.feature_column_names()
        return self.data[features].loc[t]
        
    #######################################################################
    def calculate_sma(self, window=10):
        """ Location of the close relative to the Simple Moving Average (SMA).
           This is perfectly reasonable, because e.g., if p_1,...,p_{10} are raw prices,
           then (lr_1 + lr_2 + ... + lr_{10})/10 = (log(p1/p0) + log(p2/p1) + ... + log(p_{10}/p_9))/10
                                                 = 0.1*(log(p_{10}/p0)),
            which is the mean period return over the last 10 periods.
        """
        self.data['SMA'] = self.data['close'] / self.data['close'].rolling(window=window).mean()


    #######################################################################
    def calculate_ema(self, span=10):
        """ Location of the close relative to the Exponential Moving Average (EMA)."""
        # Compute the actual EMA, but effect a scaling by recording
        #         log(most recent close / EMA),
        # instead of the raw EMA.
        # e.g., it will be negative if the stock is trading below
        # the EMA and positive if it is trading above the EMA.
        EMA = self.data['close'].ewm(span=span, adjust=False).mean()  # Actual EMA
        #self.data['lr_EMA'] = np.log(self.data['close']/EMA)
        self.data['EMA'] = self.data['close'] / EMA
        
    #######################################################################
    def calculate_macd(self, short_window=12, long_window=26, signal_window=9):
        """Calculate a relative MACD."""
        
        exp_short = self.data['close'].ewm(span=short_window, adjust=False).mean()
        exp_long = self.data['close'].ewm(span=long_window, adjust=False).mean()
        MACD_line = exp_short - exp_long
        signal_line = MACD_line.ewm(span=signal_window, adjust=False).mean()
        # What we care about is the MACD line being above or below the signal line.
        # But both of those have $ units, so we use the quotient to get a unitless quantity.
        #self.data['r_MACD'] = (MACD_line - signal_line) / self.data['close']
        
        self.data['r_MACD'] = (MACD_line - signal_line) / (0.5 * (np.abs(MACD_line) + np.abs(signal_line)))
        # Updated The MACD FORMULA

        
    #######################################################################
    def calculate_rsi(self, window=14):
        """Calculate Relative Strength Index (RSI)."""
        # This is almost exactly from Welles Wilder, except that the initial
        # arithmetic mean is replaced with an EWM.
        price_changes = self.data['close'] - self.data['close'].shift(1)
        UP = price_changes.where(price_changes>0,0)
        DOWN = -price_changes.where(price_changes<0,0)
        U = UP.ewm(alpha=1/14, adjust=False).mean()
        D = DOWN.ewm(alpha=1/14, adjust=False).mean()
        RS = U/(D+1e-10)
        self.data['RSI'] = 100 - (100/(1 + RS))
        

    #######################################################################
    def calculate_bollinger_bands(self, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        rolling_mean = self.data['close'].rolling(window).mean()
        rolling_std = self.data['close'].rolling(window).std()
        BU = rolling_mean + (rolling_std * num_std)
        BL = rolling_mean - (rolling_std * num_std)
        # Now, where is the current price relative to the upper and lower curves?
        self.data['Bollinger%b'] = (self.data['close']-BL)/(BU-BL)

    #######################################################################
    def calculate_stochastic_oscillator(self, fast_window=14, slow_window=3):
        """Calculate Stochastic Oscillator."""
        L = self.data['low'].rolling(window=fast_window).min()
        H = self.data['high'].rolling(window=fast_window).max()
        self.data['Stochastic_%K'] = 100 * ((self.data['close'] - L) / (H-L))
        self.data['Stochastic_%D'] = self.data['Stochastic_%K'].rolling(window=slow_window).mean()
    
    #######################################################################
    def calculate_fibonacci_retracement(self, window=20, levels=[0.236, 0.382, 0.5, 0.618, 0.764]):
        """Calculate Fibonacci Retracement Levels on a rolling window."""
        # I believe this is correct, but it still needs to be carefully checked.
        # The reason it's a bit tricky is that "retracement" depends on whether it has
        # recently decreased from a max to a min or increased from a min to a max.
        # If it has decreased from a max to a min, then retracement is how much relative
        # value it has regained (as a fraction).
        # If it has increased from a min to a max, then retracement is how much relative
        # value it has lost (as a fraction).
        def fib_helper_argmin(window):
            return window.argmin()
        def fib_helper_argmax(window):
            return window.argmax()

        L = self.data['low'].rolling(window=window).min()  # Could use 'low'
        H = self.data['high'].rolling(window=window).max()  # Could use 'high'
        mt = self.data['low'].rolling(window=window).apply(fib_helper_argmin, raw=True)
        Mt = self.data['high'].rolling(window=window).apply(fib_helper_argmax, raw=True)
        denom = H-L
        numer = (H-self.data['close'])*(mt < Mt) + (self.data['close'] - L)*(mt > Mt)
        R = numer/denom
        for alpha in levels:
            self.data[f'Fib_{alpha}'] = (R-alpha)/alpha
        
    #######################################################################
    def calculate_adx(self, window=14):
        """Calculate Average Directional Index (ADX), and (DI+)-(DI-)."""
        tmp_plus = self.data['high'] - self.data['high'].shift(1)
        tmp_neg = self.data['low'].shift(1) - self.data['low']
        DM_plus = tmp_plus.where(tmp_plus > tmp_neg, 0)
        DM_neg = tmp_neg.where(tmp_neg > tmp_plus, 0)

        # Inefficient: could be windowed.
        SDM_plus = pd.Series(np.nan, index=DM_plus.index)
        SDM_plus.iloc[window] = sum(DM_plus[:window])
        for t in range(window+1, len(SDM_plus)):
            SDM_plus.iloc[t] = ((window-1)/window) * SDM_plus.iloc[t-1] + DM_plus.iloc[t]

        # Again, this could be windowed.
        SDM_neg = pd.Series(np.nan, index=DM_neg.index)
        SDM_neg.iloc[window] = sum(DM_neg[:window])
        for t in range(window+1, len(SDM_neg)):
            SDM_neg.iloc[t] = ((window-1)/window) * SDM_neg.iloc[t-1] + DM_neg.iloc[t]
        

        C1 = self.data['high'] - self.data['low']
        C2 = np.abs(self.data['high'] - self.data['close'].shift(1))
        C3 = np.abs(self.data['low'] - self.data['close'].shift(1))
        TR = np.maximum(C1,C2,C3)
        # Note: TR.iloc[0] is nan, because it depends on the close from the day before.
        # Thus, the STR calculation needs to start one day later than the similar calculations above.
        # And again, this could be windowed.
        STR = pd.Series(np.nan, index=TR.index)
        STR.iloc[window+1] = sum(TR[1:window])
        for t in range(window+2, len(STR)):
            STR.iloc[t] = ((window-1)/window)*STR.iloc[t-1] + TR.iloc[t]

        DI_plus = 100*SDM_plus/STR
        DI_neg = 100*SDM_neg/STR
        DX = 100*np.abs((DI_plus - DI_neg)/(DI_plus + DI_neg))

        self.data['(DI+)-(DI-)'] = DI_plus - DI_neg
        #self.data['DI-'] = DI_neg
        self.data['ADX'] = DX.ewm(alpha=1/window, adjust=False).mean()

    #######################################################################
    def calculate_obv(self):
        """Calculate On-Balance Volume (OBV) using log returns and volume."""
        OBV = pd.Series(np.nan, index=self.data.index)
        
        obv_t = 0
        # This coud be windowed for efficiency, but it's fine:
        for i in range(1, len(self.data)):
            if self.data['lr_close'].iloc[i] > 0:
                obv_t += self.data['volume'].iloc[i]
            elif self.data['lr_close'].iloc[i] < 0:
                obv_t -= self.data['volume'].iloc[i]
            OBV.iloc[i] = obv_t

        self.data['OBV'] = OBV

    #######################################################################
    def calculate_wrobv(self, window=1200):
        """Calculate Windowed Relative On-Balance Volume (OBV) using log returns and volume."""

        OBV = pd.Series(np.nan, index=self.data.index)
        DV = pd.Series(np.nan, index=self.data.index)
        # This coud be windowed for efficiency, but it's fine:
        
        for i in range(1, len(self.data)):
            if self.data['lr_close'].iloc[i] > 0:
                DV.iloc[i] = self.data['volume'].iloc[i]
            elif self.data['lr_close'].iloc[i] < 0:
                DV.iloc[i] = -self.data['volume'].iloc[i]
            else:
                DV.iloc[i] = 0

        self.data['WROBV'] = DV.rolling(window=window).sum() / self.data['volume'].rolling(window=window).sum()


    '''
    #######################################################################
    def calculate_cci(self, window=20):
        """Calculate Commodity Channel Index (CCI) using log returns."""
        tp = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        sma = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        # Suppose window=3 and the values in that window are x=[1,2,6]. Then
        # np.abs(x-x.mean()) = [2, 1, 3], and so
        # np.mean(np.abs(x-x.mean())) = 2.
        # But, what we actually want here is (1/3)[ |1 - SMA_t| + |2 - SMA_{t+1}| + |6 - SMA_{t+2}| ].
        self.data['CCI'] = (tp - sma) / (0.015 * mad)
    '''
    #######################################################################
    def calculate_cci(self, window=20):
        """Calculate Commodity Channel Index (CCI) using log returns."""
        tp = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        sma = tp.rolling(window=window).mean()
        AD = np.abs(tp - sma)
        mad = AD.rolling(window=window).mean()
        self.data['CCI'] = (tp - sma) / (0.015 * mad)

    #######################################################################
    '''
    def calculate_ichimoku(self):
        """Calculate Ichimoku Cloud using log returns."""
        # These can be implemented, but not directly on log-returns. We'll
        # need to use rolling sums to recover cumulative values of the
        # highs and lows.
        high_9 = self.data['lr_high'].rolling(window=9).max()
        low_9 = self.data['lr_low'].rolling(window=9).min()
        high_26 = self.data['lr_high'].rolling(window=26).max()
        low_26 = self.data['lr_low'].rolling(window=26).min()
        high_52 = self.data['lr_high'].rolling(window=52).max()
        low_52 = self.data['lr_low'].rolling(window=52).min()
        
        self.data['Tenkan_sen'] = (high_9 + low_9) / 2
        self.data['Kijun_sen'] = (high_26 + low_26) / 2
        self.data['Senkou_Span_A'] = ((self.data['Tenkan_sen'] + self.data['Kijun_sen']) / 2).shift(26)
        self.data['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
        # This can be implemented, but it requires some care. It generates a signal by looking
        # at the relationship between the historical price graph and it's timeshifted value (by 26 periods);
        # specfically, it looks at how they cross and uses that to determine a signal.
        #self.data['Chikou_Span'] = self.data['lr_close'].shift(-26)
    '''
    def calculate_ichimoku(self, windowS=9, windowM=26, windowL=52):
        """Calculate Ichimoku Cloud."""
        windowS = 7
        windowM = 26
        windowL = 52
        # For minute-level data, 15, 30, 60 seems more reasonable, but I doubt it makes
        # any difference at all.
        high_S = self.data['high'].rolling(window=windowS).max()
        low_S = self.data['low'].rolling(window=windowS).min()
        high_M = self.data['high'].rolling(window=windowM).max()
        low_M = self.data['low'].rolling(window=windowM).min()
        high_L = self.data['high'].rolling(window=windowL).max()
        low_L = self.data['low'].rolling(window=windowL).min()

        tenkan = (high_S + low_S)/2
        kijun = (high_M + low_M)/2

        senkouA = (tenkan + kijun).shift(windowM)/2
        senkouB = (high_L + low_L).shift(windowM)/2

        lower = np.minimum(senkouA, senkouB)
        upper = np.maximum(senkouA, senkouB)

        below = (self.data['close'] - lower).where(self.data['close'] < lower, 0) / self.data['close']
        above = (self.data['close'] - upper).where(self.data['close'] > upper, 0) / self.data['close']
        II = (below+above)

        if True:
            self.data['II'] = II
        else:
            self.data['SENA/C'] = senkouA/self.data['close']
            self.data['SENB/C'] = senkouB/self.data['close']
