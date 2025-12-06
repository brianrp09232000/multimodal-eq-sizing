import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers# price_mlp.py – TODO: implement

######################### Select features ############################

def select_feature_columns(df: pd.DataFrame):
    """
    Decide which features to use：
    - if FEATURE_COLS is not None，use this function；
    - o/w use all numerical cols，excluding Date and ticker
    """
    global FEATURE_COLS

    if FEATURE_COLS is not None:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"FEATURE_COLS is not in the dataset: {missing}")
    else:
        exclude_cols = {DATE_COL, TICKER_COL}
        FEATURE_COLS = [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        print("Auto selected FEATURE_COLS:", FEATURE_COLS)

    if TARGET_COL not in FEATURE_COLS:
        raise ValueError(
            f"TARGET_COL='{TARGET_COL}'must included in the FEATURE_COLS，"
        )

    return FEATURE_COLS



############################### Min-max scale ##############################

def scale_features(df: pd.DataFrame, feature_cols):
    """
    data normalization: MinMax Scale
    """
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df[feature_cols].values)

    scaled_df = df.copy()
    scaled_df[feature_cols] = scaled_array
    return scaled_df, scaler


################# Creating a slide window for a sticker (use previous [window size] days' features for prediction)###############


def create_sliding_windows_single_stock(
    stock_df: pd.DataFrame,
    feature_cols,
    target_col,
    window_size: int
):
    """
    Input: a ticker's time sequence data (sorted by time and normalized）
    Output: X_i, y_i
    - X_i: (n_samples_i, window_size, num_features)
    - y_i: (n_samples_i,)
    """
    data = stock_df[feature_cols].values
    target_index = feature_cols.index(target_col)

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size, :])                  
        y.append(data[i + window_size, target_index])          
    X = np.array(X)
    y = np.array(y)

    return X, y



################## split train/val/test #########################

def split_single_stock(X, y, test_ratio, val_ratio):
    """
    Split by time for a single ticker:
    Use the earliest (1‒test_ratio) portion as train + validation, and the most recent test_ratio portion as test.
    Then split trainval again so that val_ratio of it becomes the validation set.
    """
    n = len(X)
    if n == 0:
        return (None,) * 6

    test_size = int(n * test_ratio)
    trainval_size = n - test_size

    X_trainval = X[:trainval_size]
    y_trainval = y[:trainval_size]

    X_test = X[trainval_size:]
    y_test = y[trainval_size:]

    val_size = int(trainval_size * val_ratio)
    X_train = X_trainval[:-val_size]
    y_train = y_trainval[:-val_size]
    X_val   = X_trainval[-val_size:]
    y_val   = y_trainval[-val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_panel_dataset(
    df_scaled: pd.DataFrame,
    feature_cols,
    target_col,
    window_size,
    test_ratio,
    val_ratio
):
    """
    For each stock in the df:
    Construct sliding windows.
    Split into train/validation/test.
    Finally, combine samples from all stocks into one dataset.
    """
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    X_test_list,  y_test_list  = [], []

    tickers = df_scaled[TICKER_COL].unique()
    print(f"Total tickers: {len(tickers)}")

    for tic in tickers:
        sub = df_scaled[df_scaled[TICKER_COL] == tic].copy()
        sub = sub.sort_values(DATE_COL)

        X, y = create_sliding_windows_single_stock(
            sub, feature_cols, target_col, window_size
        )

        if len(X) == 0:
            continue

        (X_tr, y_tr,
         X_val, y_val,
         X_te, y_te) = split_single_stock(X, y, test_ratio, val_ratio)

        if X_tr is None:
            continue

        if len(X_tr):
            X_train_list.append(X_tr)
            y_train_list.append(y_tr)
        if len(X_val):
            X_val_list.append(X_val)
            y_val_list.append(y_val)
        if len(X_te):
            X_test_list.append(X_te)
            y_test_list.append(y_te)

    # concatenate all
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val   = np.concatenate(X_val_list, axis=0) if X_val_list else np.empty((0, WINDOW_SIZE, len(feature_cols)))
    y_val   = np.concatenate(y_val_list, axis=0) if y_val_list else np.empty((0,))
    X_test  = np.concatenate(X_test_list, axis=0)
    y_test  = np.concatenate(y_test_list, axis=0)

    print("Final shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val  :", X_val.shape,   "y_val  :", y_val.shape)
    print("  X_test :", X_test.shape,  "y_test :", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test



####################### Build hybrid GRU LSTM model ###################

def build_hybrid_gru_lstm(input_shape):
    """
    input_shape: (WINDOW_SIZE, num_features)

    Input
      → GRU(128, return_sequences=True) + Dropout(0.2)
      → LSTM(128) + Dropout(0.2)
      → Dense(64, relu)
      → Dense(1, linear)
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.GRU(
        GRU_UNITS,
        return_sequences=True,
        dropout=DROPOUT_RATE
    )(inputs)

    x = layers.LSTM(
        LSTM_UNITS,
        dropout=DROPOUT_RATE
    )(x)

    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_GRU_LSTM")

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_hybrid_gru_lstm2(input_shape):
    """
    input_shape: (WINDOW_SIZE, num_features)

    Input
      → GRU(128, return_sequences=True) + Dropout(0.2)
      → LSTM(128) + Dropout(0.2)
      → Dense(64, relu)
      → Dense(1, linear)
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.GRU(
        128,
        return_sequences=True,
        dropout=DROPOUT_RATE
    )(inputs)

    x = layers.GRU(
        64,
        return_sequences=True,
        dropout=DROPOUT_RATE
   )(x)

    x = layers.LSTM(
        64,
        return_sequences=True,
        dropout=DROPOUT_RATE
    )(x)

    x = layers.LSTM(
        32,
        dropout=DROPOUT_RATE
    )(x)

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_GRU_LSTM")

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model


################ Evaluation ############################

# plot diagnostic learning curves
def summarize_price_model_diagnostics(model, history):
  """plots price model loss and accuracy curves of the train and validation data
  inputs: model - tensorflow model; history - model.fit output
  outputs: none"""
  fig, ax = plt.subplots(1,2, figsize=(20, 10))
  # plot loss
  ax[0].set_title(model.name+': Loss Curves', fontsize=20)
  ax[0].plot(history.history['loss'], label='train')
  ax[0].plot(history.history['val_loss'], label='validation')
  ax[0].set_xlabel('Epochs', fontsize=15)
  ax[0].set_ylabel('Loss', fontsize=15)
  ax[0].legend(fontsize=15)
  # plot mae
  ax[1].set_title(model.name+': MAE Curves', fontsize=20)
  ax[1].plot(history.history['mae'], label='train')
  ax[1].plot(history.history['val_mae'], label='validation')
  ax[1].set_xlabel('Epochs', fontsize=15)
  ax[1].set_ylabel('MAE', fontsize=15)
  ax[1].legend(fontsize=15)
  
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
