"""Module that implements the trading manager class"""

import asyncio

import numpy as np

from project_logging import lib_logger as logger
from utils import get_intervals, get_symbols, model_hash


class TradeManager:
    def __init__(self, session, database, config, live_market, predictions):
        self.session = session
        self.db = database
        self.cfg = config
        self.market = live_market
        self.predictions = predictions
        self.symbols = get_symbols(self.cfg)
        self.intervals = get_intervals(self.cfg)
        logger.debug("TradeManager successfully initialized.")

    async def trade_pairs(self):
        """Consumes live market data and predictions to manage trades"""
        try:
            while True:
                if self.market:
                    for symbol, interval in zip(self.symbols, self.intervals):
                        model_name = model_hash(symbol, interval)
                        if not (
                            pred := self.predictions.get(model_name, np.array([]))
                        ).size:
                            continue
                        logger.info(
                            f"({model_name}) - "
                            f"Current price: {self.market[symbol]['k']['c']}"
                            f"  |  Prediction: {pred}"
                        )
                await asyncio.sleep(5)
        except asyncio.exceptions.CancelledError:
            logger.info("CTRL + C detected, closing TradeManager.")
