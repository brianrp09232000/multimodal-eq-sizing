# price_mlp.py – TODO: implement
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, GRU

def simple_rnn_price_model(model_shape):
    model = Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=model_shape))
    model.add(SimpleRNN(units=50, return_sequences=True))
    model.add(SimpleRNN(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
  


def gru_price_model(model_shape):
    model = Sequential()
    model.add(GRU(units=128, dropout=0.2, activity_regularizer='l2', return_sequences=True,
              input_shape=model_shape))
    model.add(GRU(units=64, dropout=0.2, activity_regularizer='l2', return_sequences=True,
              input_shape=model_shape))
    model.add(GRU(units=32, dropout=0.2, activity_regularizer='l2'))
    model.add(Dense(1))
    return model
  