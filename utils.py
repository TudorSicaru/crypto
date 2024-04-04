"""Module that implements utils functions for the project"""

import configparser
from datetime import datetime, timezone
from pathlib import Path
from typing import List

INTERVALS_TO_MILISECONDS = {
    "1m": 60000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "30m": 1800000,
    "1h": 3600000,
    "2h": 7200000,
    "4h": 14400000,
    "6h": 21600000,
    "8h": 28800000,
    "12h": 43200000,
    "1d": 86400000,
    "3d": 259200000,
    "1w": 604800000,
    "1M": 2630016000,  # Average number of sesconds in a month
}


def get_symbols(config: configparser.ConfigParser) -> List[str]:
    """Get the symbol from config and clean up any
    empty string, etc
    """
    return [pair.strip().upper() for pair in config["trading"]["symbols"].split(",")]


def get_intervals(config: configparser.ConfigParser) -> List[str]:
    """Parses the config to retrieve the intervals
    duration that will be fetched from binance API
    """
    return [interval.strip() for interval in config["trading"]["intervals"].split(",")]


def get_limit(config: configparser.ConfigParser) -> List[int]:
    """Get the number of intervals that will be fetched from binance

    NOTE: A maximum of 1000 intervals can be fetched
    """
    return [
        int(interval.strip())
        for interval in config["trading"]["interval_count"].split(",")
    ]


def miliseconds_to_utc(timestamp):
    """Converts a Binance timestamp (in ms)
    to a datetime that is timezone aware"""
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)


def utc_to_miliseconds(utc_dt):
    """Receives an AWARE UTC datetime object and converts it to timestamp"""
    if not isinstance(utc_dt, datetime):
        raise ValueError(
            f"The `utc_dt` ({utc_dt}) argument should be datetime object. "
        )
    return int(utc_dt.timestamp() * 1000)


def model_hash(symbol, interval, timestamp=None):
    """Builds a unique model `name`.

    NOTE: timestamp not used for the moment
    """
    return f"{symbol}_{interval}"


def model_filename(model_name):
    """Creates the filename used to load/save the model."""
    return Path("trained_models") / f"pre_trained_lstm_{model_name}.keras"


def scaler_filenames(model_name):
    """Creates the filename used to load/save the scalers."""
    x_scaler = Path("trained_models") / f"x_scaler_{model_name}.joblib"
    y_scaler = Path("trained_models") / f"y_scaler_{model_name}.joblib"
    return x_scaler, y_scaler
