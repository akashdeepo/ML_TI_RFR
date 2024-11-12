# pred_nn.py

import pandas as pd
import numpy as np
from stockdata import StockData
import tensorflow as tf
import matplotlib.pyplot as plt

# I just copy/pasted this code in from another project, but haven't finished testing it yet,
# although I've made some modifications.

"""
  Internally, this class will scale the Y values to mean 0, std 1. But the
  apply() and get_signal() methods will undo that scaling. The optimization
  just proceeds much more reliably with this scaling.
"""


##############################################################################
class NN_Predictor: 
    def __init__(self, name, dense_arch=[20,50,50,10],
                 train_epochs=1000,
                 validation_frac=0.15,
                 sell_quantile=1/3, buy_quantile=2/3,
                 monitor_val_loss = True,
                 plot_loss = False,
                 verbose=2):
        self.name = name
        self.arch = dense_arch
        self.epochs = train_epochs
        self.verbose = verbose
        self.model = None
        self.sell_quantile = sell_quantile
        self.buy_quantile = buy_quantile
        self.validation_frac = validation_frac
        self.monitor_val_loss = monitor_val_loss
        self.plot_loss = plot_loss

    def get_name(self):
        return self.name

    def get_quantiles(self):
        return self.sell_quantile, self.buy_quantile
      
    def train(self, trainX, trainY):
        VALIDATION_SET = 'random' # Either 'random' or 'tail'.

        # Make a copy of trainY, because we're going to alter it:
        Y = trainY.copy()
        self.y_mean = Y.mean()
        self.y_std = Y.std()
        Y = (Y - self.y_mean)/self.y_std
        
        
        self.num_features = trainX.shape[1]
        self._build_model(trainX, trainY)
        if self.verbose>0:
            print(f"*** Fitting model on trainX, shape={trainX.shape}")

        N = len(trainX)
        if VALIDATION_SET=='random':
            val_size = int(self.validation_frac * N)
            val_ind = np.sort(np.random.choice(N, val_size,replace=False))
            tr_ind = np.delete(np.arange(N), val_ind)
            #trainX = trainX.to_numpy()
            
            valX = trainX.iloc[val_ind]
            valY = Y.iloc[val_ind]
            valData = (valX, valY)
            trX = trainX.iloc[tr_ind]
            trY = Y.iloc[tr_ind]
        elif VALIDATION_SET=='tail':
            val_size = int(self.validation_frac*N)
            valX, valY = trainX[-val_size:], Y.iloc[-val_size:]
            trX, trY = trainX[:-val_size], Y.iloc[:-val_size]
            valData = (valX, valY)
        else:
            valData = None
            trX = trainX
            trY = Y
            
        # This is absurdly overfitting the data, but it gives results comparable to the RFR:
        #val_loss_monitor = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=40, restore_best_weights=True)
        # Something like this would be more reasonable:
        val_loss_monitor = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                                            restore_best_weights=False,
                                                            start_from_epoch=5)
        # This one is a custom callback defined below. I just wanted to do a quick experiment with a different
        # early stopping condition:
        #loss_monitor = EarlyStoppingAtMinCombined(patience=20)
        # But I think what we should do is write a custom monitor, and continue training while the validation loss is
        # either improving or (approximately) less than the training loss.

        if self.monitor_val_loss:
            CB = [val_loss_monitor]
        else:
            CB = []
        history = self.model.fit(trX, trY,
                       epochs=self.epochs,
                       shuffle=True,
                       validation_data=valData,
                       callbacks = CB,
                       verbose=self.verbose)
        self.train_history = history.history
        Y = self.apply(trainX)
        self.bins = [np.quantile(Y, self.sell_quantile), np.quantile(Y, self.buy_quantile)]
        print(f"NN_Predictor(): bin cutoffs={self.bins}")

        if self.plot_loss:
            fig, ax = plt.subplots()
            x = np.arange(len(self.train_history['loss']))
            ax.set_yscale('log')
            ax.plot(x, self.train_history['loss'], label='loss')
            ax.plot(x, self.train_history['val_loss'], label='val_loss')
            ax.legend()
            plt.show()
        

  ##########################################################################
    def _build_model(self, trainX, trainY):

        if not(self.model is None):
            del self.model # Pointless attempt to avoid TensorFlow (or Keras) memory leak.

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(self.num_features,)))
        
        # The following normalization layer will scale inputs so that each
        # input of the training set has expected value 0, expected std 1.
        normalization_layer = tf.keras.layers.Normalization()
        normalization_layer.adapt(trainX.to_numpy())
        self.model.add(normalization_layer)

        for n in self.arch:
            LeakyR = tf.keras.layers.LeakyReLU(negative_slope=0.1)
            self.model.add(tf.keras.layers.Dense(n, activation=LeakyR))

        #self.model.add(tf.keras.layers.Dense(1, activation=self.LeakyR))
        self.model.add(tf.keras.layers.Dense(1, activation=None))
        if self.verbose>0:
            print(f"{self.num_features} --> {self.arch} --> 1 model built.")

        self.model.compile(optimizer='adam', loss="mse")


    
    def apply(self, X):
        pass
        # return the raw result of applying this model to X,
        # which should be a DataFrame.
        y = self.model.predict(X).reshape(-1)
        Y = y*self.y_std + self.y_mean
        return pd.Series(Y, index=X.index)
      
    def get_signal(self, Xdf):
        # X : An ndarray of shape (N,n), where X[0],X[1],...,X[N-1] are
        #     feature values.
        # Return value: An ndarray Y of shape (N,k), where k is the number
        #     of categories and Y[j] is a vector describing the `likelihood'
        # that sample j belongs to each corresponding category.
        # method:  "cat" ==> catgorical output only
        #          "prob" ==> output categorical 'likelihood values' (a vector of those, for each input)
        Y = self.apply(Xdf).to_numpy()
        X = Xdf.to_numpy()
        N = X.shape[0]
        
        #Y = self.model(X).numpy()
        Y_discretized = np.digitize(Y, self.bins)
        retval = []
        sig = ["sell", "hold", "buy"]
        for i in range(Y_discretized.shape[0]):
            retval.append(sig[Y_discretized[i]])
        return pd.Series(retval, index=Xdf.index)

########################################################
# A custom callback; I copy/pasted from https://keras.io/guides/writing_your_own_callbacks/
# and modified for some experiments.
class EarlyStoppingAtMinCombined(tf.keras.callbacks.Callback):
    """Stop training when val_loss + loss stops improving. This is very ad-hoc,
       and mostly just as an example. When I have time, I'll sit down to do the
       math and figure out if there is some reasonable tradeoff that's worth trying.
       But I strongly suspect that there isn't - I think that probably the best thing
       to do is minimize the validation loss itself. But it's not obvious - consider this:
         In the end, what we really care about is the strength of the correlation between
       predictions and true values (outside of the training set). So perhaps we should
       compute that correlation on the validation set and use that to determine a
       stopping condition. I suspect it won't give results that are any better on the
       test set, but it's worth a shot. But it could. The idea is that we don't really
       care about overestimates of large values or underestimates of small values
       because they will still send the right sell/hold/buy signal.
         But that's also why I want to try a classifier instead. I think categorical
       cross entropy makes total sense here.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss") + logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")

        
