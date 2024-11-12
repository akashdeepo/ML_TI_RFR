# pred_nnc.py

import pandas as pd
import numpy as np
from stockdata import StockData
import tensorflow as tf

# This code isn't going to be used in the current project, but I do think that
# a classifier has a good chance to outperform regression so I will eventually
# flush this out to a working state and complete the 'apply' method to use the
# 3-node output to construct numerical estimates.
#   But in the meantime, don't use this class for anything - it won't work at all!


##############################################################################
class NNC_Predictor: 
    def __init__(self, name, sell_quantile=0.33, buy_quantile=0.66, dense_arch=[20,50,50,10], train_epochs=1000):
        self.name = name
        self.sell_quantile = sell_quantile
        self.buy_quantile = buy_quantile
        self.arch = dense_arch
        self.epochs = train_epochs
        self.verbose = 2
        self.model = None
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.loss_fn_logits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.LeakyR = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.model = None

    def get_name(self):
        return self.name

    def get_quantiles(self):
        return self.sell_quantile, self.buy_quantile

    def train(self, trainX, trainY):
        self.cats = [0,1,2]
        self.num_cats = 3
        self.num_features = trainX.shape[1]
        self._build_model()
        if self.verbose>0:
            print(f"*** Fitting model on trainX, shape={trainX.shape}")
        bins = [np.quantile(trainY, self.sell_quantile),
                np.quantile(trainY, self.buy_quantile)]
        Y_discretized = np.digitize(trainY, bins)
        """
        history = self.model.fit(trainX, Y_discretized,
                       epochs=self.epochs,
                       shuffle=True,
                       verbose=self.verbose)
        """
        N = len(trainX)
        val_ind = np.sort(np.random.choice(N, int(0.1*N),replace=False))
        tr_ind = np.delete(np.arange(N), val_ind)
        trainX = trainX.to_numpy()
        
        valX = trainX[val_ind]
        valY = Y_discretized[val_ind]
        trX = trainX[tr_ind]
        trY = Y_discretized[tr_ind]
        history = self.model.fit(trX, trY,
                       epochs=self.epochs,
                       shuffle=True,
                       validation_data=(valX,valY),
                       verbose=self.verbose)
        print(history.history['val_accuracy'])

  ##########################################################################
    def _build_model(self):
        # SOFTMAX = True uses softmax activation function at the output nodes.
        #           False uses LeakyR.
        SOFTMAX = True

        if not(self.model is None):
            del self.model # Pointless attempt to avoid TensorFlow (or Keras) memory leak.

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(self.num_features,)))
        # The following normalization layer will scale inputs so that each
        # input of the training set has expected value 0, expected std 1.
        # nlayer = tf.keras.layers.Normalization(axis=-1)
        # nlayer.adapt(...)
        # self.model.add(nlayer)
        for n in self.arch:
            self.model.add(tf.keras.layers.Dense(n, activation='softplus'))

        if SOFTMAX:
            self.model.add(tf.keras.layers.Dense(self.num_cats, activation='softmax'))
            loss_fn = self.loss_fn
        else:
            self.model.add(tf.keras.layers.Dense(self.num_cats, activation=self.LeakyR))
            loss_fn = self.loss_fn_logits
        if self.verbose>0:
            print(f"{self.num_features} --> {self.arch} --> {self.num_cats} model built.")

        self.model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])


    
    def apply(self, X):
        pass
        # return the raw result of applying this model to X,
        # which should be a DataFrame.
        return pd.Series(self.model.predict(X), index=X.index)
      
    def get_signal(self, Xdf):
        # X : An ndarray of shape (N,n), where X[0],X[1],...,X[N-1] are
        #     feature values.
        # Return value: An ndarray Y of shape (N,k), where k is the number
        #     of categories and Y[j] is a vector describing the `likelihood'
        # that sample j belongs to each corresponding category.
        # method:  "cat" ==> catgorical output only
        #          "prob" ==> output categorical 'likelihood values' (a vector of those, for each input)
        X = Xdf.to_numpy()
        N = X.shape[0]
        k = max(self.cats)+1

        Y = self.model(X).numpy()

        c = Y.argmax(axis=1)
        retval = []
        sig = ["sell", "hold", "buy"]
        for i in range(c.shape[0]):
            retval.append(sig[c[i]])
        return pd.Series(retval, index=Xdf.index)

        
