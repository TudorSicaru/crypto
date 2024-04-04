import asyncio
from configparser import ConfigParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from asyncdb import AsyncDB
from project_logging import train_logger as logger
from utils import (
    get_intervals,
    get_symbols,
    model_filename,
    model_hash,
    scaler_filenames,
)


class TrainingError(Exception):
    pass


def create_sequences(data, time_steps, n_outputs):
    """Creates multidimensional arrays used as inputs (X) and outputs (y) for
    training a LSTM model.

    NOTE: The length of each array will be len(data) - time_steps - n_outputs
    which is usually 999 - 30 - 5  (given 30 candles as input, we predict next 5)
    As per above example, X consists of a sequence of 964 arrays, each array
    containing 30 candles.
    The 'y' array consists of a sequence of arrays, each array representing the
    5 close_prices for future candles.

    We basically give as input
    X[0] = [
            [open_price0, high0, low0, close_price0, volume0, number_of_trades0]
            ...
            [open_price29, high29, low29, close_price29, volume29, number_of_trades29]
            ]
    and we train to return:
    y[0] = [close_price30, close_price31, close_price32, close_price33, close_price34]

    X and y will be collections of 964 such elements (when len(data)== 999)
    """
    X, y = [], []
    for i in range(len(data) - time_steps - n_outputs):
        sequence = data[i : i + time_steps]
        result = data[i + time_steps : i + time_steps + n_outputs]
        result = [row[3] for row in result]
        X.append(sequence)
        y.append(result)
    return np.array(X), np.array(y)


def init_scalers(training_data, model_name, models):
    """Initialize scalers and `manually` compute the price_range, volume_range,
    trades_range for scaling. This is because we will work with and predict
    values that might land outside the range of training data, messing with the
    scaler and prediction outcomes.

    NOTE: I need to experiment if this is needed, the predictions might work just
        fine without custom scalers, they will just return values > 1 that need to
        be scaled back up, hopefully to correct numbers, because the MinMaxScaler
        is meant to scale in the interval (0,1).
    """
    # Initialize 2 scalers for input/output data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # Create a range for prices
    price_range = [
        0.85 * float(training_data["low"].min()),
        1.15 * float(training_data["high"].max()),
    ]

    # Create a range for volumes
    volume_range = [
        0.85 * float(training_data["volume"].min()),
        1.15 * float(training_data["volume"].max()),
    ]

    # Create a range for trades
    trades_range = [
        0.85 * float(training_data["number_of_trades"].min()),
        1.15 * float(training_data["number_of_trades"].max()),
    ]
    logger.debug(
        f"\ninit_scalers - Price range: {price_range}"
        f"\ninit_scalers - Volume range: {volume_range}"
        f"\ninit_scalers - Trades range: {trades_range}"
    )

    # Calculate scales such that the min price is close to 0 and max close to 1
    price_scale = 1 / (price_range[1] - price_range[0])
    min_price_scale = -price_range[0] * price_scale

    volume_scale = 1 / (volume_range[1] - volume_range[0])
    min_vol_scale = -volume_range[0] * volume_scale

    trades_scale = 1 / (trades_range[1] - trades_range[0])
    min_trades_scale = -trades_range[0] * trades_scale

    # Set the attributes of x_scaler
    x_scaler.scale_ = np.array([price_scale] * 4 + [volume_scale] + [trades_scale])
    x_scaler.min_ = np.array(
        [min_price_scale] * 4 + [min_vol_scale] + [min_trades_scale]
    )
    x_scaler.data_min_ = np.array(
        [price_range[0]] * 4 + [volume_range[0]] + [trades_range[0]]
    )
    x_scaler.data_max_ = np.array(
        [price_range[1]] * 4 + [volume_range[1]] + [trades_range[1]]
    )
    x_scaler.data_range_ = x_scaler.data_max_ - x_scaler.data_min_
    x_scaler.feature_range = (0, 1)

    # Set the attributes of y_scaler
    y_scaler.scale_ = np.array([price_scale])
    y_scaler.min_ = np.array([min_price_scale])
    y_scaler.data_min_ = np.array([price_range[0]])
    y_scaler.data_max_ = np.array([price_range[1]])
    y_scaler.data_range_ = y_scaler.data_max_ - y_scaler.data_min_
    y_scaler.feature_range = (0, 1)

    # Add the scaler to the model dictionary for ease of access
    models[model_name]["x_scaler"] = x_scaler
    models[model_name]["y_scaler"] = y_scaler


def prepare_data(
    model_name, training_data, time_steps, n_outputs, train_factor, models
):
    """Model that prepares the data for training"""
    training_data = pd.DataFrame([dict(record) for record in training_data])
    if training_data.empty:
        raise TrainingError(
            f"The archive is empty, can't train the model at the moment"
        )
    X, y = create_sequences(training_data.values, time_steps, n_outputs)
    train_size = int(len(X) * train_factor)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    # Create scalers for this model
    init_scalers(training_data, model_name, models)
    # Flatten the X data so we can fit/transform
    samples, timesteps, features = X_train.shape
    test_samples, test_timesteps, test_features = X_test.shape
    X_train = X_train.reshape(-1, features)
    X_test = X_test.reshape(-1, features)
    # Scale the training and test data
    X_train = models[model_name]["x_scaler"].transform(X_train)
    X_test = models[model_name]["x_scaler"].transform(X_test)
    y_train = models[model_name]["y_scaler"].transform(y_train)
    y_test = models[model_name]["y_scaler"].transform(y_test)
    # Restore X_train and X_test to 3-d arrays
    X_train = X_train.reshape(samples, timesteps, features)
    X_test = X_test.reshape(test_samples, test_timesteps, test_features)
    return X_train, X_test, y_train, y_test


def init_model(symbol, interval, time_steps, features, n_outputs, models):
    """Instantiates an untrained model"""
    model_name = model_hash(symbol, interval)
    logger.info(f"Initializing LSTM model for {model_name}")

    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(time_steps, features)),
            tf.keras.layers.LSTM(100, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(100, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_outputs),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    # Show a summary of the model
    model.summary()
    # Add the untrained model to the models dict
    models[model_name] = {"obj": model, "trained": False}


def train_model(model_name, X_train, X_test, y_train, y_test, models):
    """Attempts to train a model for the given symbol/interval combination"""
    try:
        if model_name not in models:
            raise TrainingError(f"Model for {model_name} was not found.")
        if models[model_name]["trained"]:
            raise TrainingError(f"Model was already trained. Skip training.")
        logger.info(f"Start training LSTM model for {model_name}")
        # Callback to stop training if the model's val_loss increases
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )
        # Actual training
        _ = models[model_name]["obj"].fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=64,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[early_stopping],
        )
        models[model_name]["trained"] = True
        models[model_name]["obj"].save(model_filename(model_name))
        x_name, y_name = scaler_filenames(model_name)
        joblib.dump(models[model_name]["x_scaler"], x_name)
        joblib.dump(models[model_name]["y_scaler"], y_name)
    except TrainingError as err:
        logger.warning(
            f"Failed to train the LSTM model {model_name}. Reason:\n{repr(err)}"
        )
        models[model_name]["trained"] = False


async def main():
    cfg = ConfigParser()
    cfg.read("config.ini")
    models = {}
    time_steps = int(cfg["machine_learning"]["input_candles"])
    features = int(cfg["machine_learning"]["input_features"])
    n_outputs = int(cfg["machine_learning"]["output_candles"])
    train_factor = 0.8
    async with AsyncDB(cfg) as database:
        for symbol, interval in zip(get_symbols(cfg), get_intervals(cfg)):
            model_name = model_hash(symbol, interval)
            # Initialize model
            init_model(symbol, interval, time_steps, features, n_outputs, models)
            # Fetch training data
            training_data = await database.fetch_archive_data(symbol, interval)
            # Prepare training data
            X_train, X_test, y_train, y_test = prepare_data(
                model_name, training_data, time_steps, n_outputs, train_factor, models
            )
            # Train the model
            train_model(model_name, X_train, X_test, y_train, y_test, models)
            # Print predict(X_test) vs (y_test), which is predict vs actual
            for input_data, expected_output in zip(X_test, y_test):
                pred = models[model_name]["obj"].predict(
                    input_data.reshape(1, time_steps, features)
                )
                pred = models[model_name]["y_scaler"].inverse_transform(
                    pred.reshape(n_outputs, 1)
                )
                actual = models[model_name]["y_scaler"].inverse_transform(
                    expected_output.reshape(n_outputs, 1)
                )
                input_data = input_data.reshape(-1, features)
                input_data = models[model_name]["x_scaler"].inverse_transform(
                    input_data
                )
                logger.warning(f"Input Data[-1]: {input_data[-1][3]}")
                logger.info(f"Predicted:         {pred.reshape(1, n_outputs)}")
                logger.debug(f"Actual:           {actual.reshape(1, n_outputs)}")


if __name__ == "__main__":
    asyncio.run(main())
