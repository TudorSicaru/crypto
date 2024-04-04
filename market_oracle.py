"""Module that implements the MarketOracle - a LSTM model that generates
price predictions for cryptocurrencies.
"""

import asyncio

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from project_logging import lstm_logger as logger
from utils import (
    get_intervals,
    get_symbols,
    model_filename,
    model_hash,
    scaler_filenames,
)


class OracleException(Exception):
    pass


class MarketOracle:
    def __init__(self, database, config, predictions):
        self.cfg = config
        self.db = database
        self._models = {}
        self._scalers = {}
        self.symbols = get_symbols(self.cfg)
        self.intervals = get_intervals(self.cfg)
        self.poll_time = int(self.cfg["machine_learning"]["poll_time"])
        self.time_steps = int(self.cfg["machine_learning"]["input_candles"])
        self.features = int(self.cfg["machine_learning"]["input_features"])
        self.outputs = int(self.cfg["machine_learning"]["output_candles"])
        self.predictions = predictions
        self._init_models()
        logger.debug("MarketOracle was initialized.")

    def _init_models(self):
        """Function that is supposed to be called once at the initialization
        of this class. It will go through all symbols/intervals we are currently
        trading and initialize models to predict prices for each combination
        """
        logger.info("MarketOracle - attempting to load all models from file.")
        for symbol, interval in zip(self.symbols, self.intervals):
            model_name = model_hash(symbol, interval)
            self._load_model_from_file(model_name=model_name)

    def _load_model_from_file(self, model_name):
        """Attempts to load the pre-trained model from file."""
        model = {}
        model_file = model_filename(model_name)
        x_name, y_name = scaler_filenames(model_name)
        load_conditions = [model_file.exists(), x_name.exists(), y_name.exists()]
        if all(load_conditions):
            logger.info(f"Loading pre-trained LSTM model for {model_name} from file")
            model["obj"] = tf.keras.models.load_model(model_file)
            model["x_scaler"] = joblib.load(x_name)
            model["y_scaler"] = joblib.load(y_name)
            model["obj"].summary()
            self._models[model_name] = model

    async def predict_pairs(self):
        """Runs concurrent `predict` tasks
        for each symbol-interval combination"""
        try:
            tasks = []
            for symbol, interval in zip(self.symbols, self.intervals):
                tasks.append(self.predict_pair(symbol, interval))
            await asyncio.gather(*tasks)
        except asyncio.exceptions.CancelledError:
            logger.info("CTRL + C detected, closing MarketOracle.")
            # TODO: Implement logic to save the oracle training data to a new file

    async def predict_pair(self, symbol, interval):
        model_name = model_hash(symbol, interval)
        while True:
            await asyncio.sleep(5)
            input_data = await self.db.fetch_archive_data(symbol, interval)
            # Taking only time_steps klines from the archive, ordered by time ASC
            input_data[:] = input_data[-self.time_steps :]
            prediction = self.predict(model_name, input_data)
            self.predictions[model_name] = prediction

    def predict(self, model_name, input_data):
        """Returns a prediction using the model associated with the symbol/interval
        we are predicting."""
        if model_name not in self._models:
            logger.warning(
                f"No model associated with {model_name}. You need to train one first."
            )
            return np.array([])
        model = self._models[model_name]["obj"]
        input_data = pd.DataFrame([dict(record) for record in input_data]).values
        # Scale the data
        transformed = self._models[model_name]["x_scaler"].transform(input_data)
        # Convert the transformed array to a 3-d array
        input_array = transformed.reshape(1, self.time_steps, self.features)
        # Reshape
        prediction = model.predict(input_array)
        return (
            self._models[model_name]["y_scaler"]
            .inverse_transform(prediction.reshape(self.outputs, 1))
            .reshape(1, self.outputs)
        )
