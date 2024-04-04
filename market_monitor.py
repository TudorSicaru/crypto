"""Module that implements the market monitor class using the binance API"""

import asyncio
import contextlib
from datetime import datetime, timezone

from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException

from project_logging import lib_logger as logger
from utils import (
    INTERVALS_TO_MILISECONDS,
    get_intervals,
    get_symbols,
    miliseconds_to_utc,
)


@contextlib.asynccontextmanager
async def async_binance_client(config):
    """Implements an async context manager to ensure we close the conn
    once an exception is caught"""
    api_key = config["binance"]["api_key"]
    api_secret = config["binance"]["api_secret"]
    client = await AsyncClient.create(api_key, api_secret)
    try:
        yield client
    finally:
        logger.info("Closing AsyncClient connection to Binance")
        await client.close_connection()


class MarketMonitor:
    """Class that implements monitoring methods in order to fetch
    live and historical data from the market using the Binance async client
    """

    def __init__(self, client, database, config, live_market):
        """Sets MarketMonitor attributes

        Params:
            client (binance.AsyncClient): Binance async client.
            config (ConfigParser): ConfigParser initialized with config.ini.
            live_market (dict): Dictionary used to store live market data.
        """
        self.client = client
        self.db = database
        self.cfg = config
        self.market = live_market
        logger.debug(f"MarketMonitor initialized.")

    async def track_klines(self):
        """Async method to track candles for all symbols from config.ini and
        to manage the archives per symbol, concurrently."""
        tasks = []
        for symbol, interval in zip(get_symbols(self.cfg), get_intervals(self.cfg)):
            tasks.append(self.track_single_kline(symbol, interval))
            tasks.append(self.manage_kline_archive(symbol, interval))
        logger.info(f"MarketMonitor - track_klines starting tasks.")
        await asyncio.gather(*tasks)

    async def track_single_kline(self, symbol, interval):
        """Tracks `live` kline data for a symbol using the async kline_socket"""
        logger.debug(f"Listening for live kline data for `{symbol}`")
        bsm = BinanceSocketManager(self.client)
        try:
            async with bsm.kline_socket(symbol, interval=interval) as trade_stream:
                while True:
                    self.market[symbol] = await trade_stream.recv()
        except OSError as err:
            logger.warning(
                f"Caught exception {repr(err)}. Closing kline socket for {symbol}."
            )

    async def manage_kline_archive(self, symbol, interval):
        """Keeps an eye on current market data for the symbol. When a
        new kline starts, it will archive the previous kline(s), closing
        any gap between the archive and live_market data.

        NOTE: The current/last kline is not stored, because it's
            ongoing and thus incomplete.
        """
        incomplete_kline = -1
        poll_time = int(self.cfg["archive"]["poll_time"])
        logger.info(
            f"Archive monitoring for `{symbol}` will "
            f"be polling every {poll_time} seconds."
        )
        while True:
            await asyncio.sleep(poll_time)
            try:
                if not (
                    start_time := await self.new_kline_started(
                        symbol,
                        interval,
                    )
                ):
                    logger.info(
                        "Nothing new to archive. "
                        f"Checking again in {poll_time} seconds."
                    )
                    continue
                klines = await self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_time,
                    limit=1000,
                )
                if not klines[:incomplete_kline]:
                    continue
                await self.db.archive_historical_data(
                    symbol, interval, klines[:incomplete_kline]
                )
            except (KeyError, IndexError) as err:
                logger.warning(f"Got exception for {symbol}: {repr(err)}")
            except (BinanceAPIException, BinanceRequestException) as err:
                logger.error(
                    f"Found exception with status code {err.status_code}:\n{repr(err)}"
                )

    async def new_kline_started(self, symbol, interval):
        """Determines if we have new candles based on latest
        open_time from the database and the latest kline_time
        reported by the live market (through the websocket)

        Returns: The start time to use in the query that fetches
            the new klines. INT(0) if we don't expect new klines
        """
        # Get the unit of 1 kline in miliseconds
        kline_miliseconds = INTERVALS_TO_MILISECONDS[interval]
        max_klines = kline_miliseconds * int(self.cfg["trading"]["max_archive_klines"])
        # Get current timestamp (miliseconds)
        now_miliseconds = datetime.now(tz=timezone.utc).timestamp() * 1000
        # Get archive timestamp
        archive_timestamp = await self.db.get_latest_archive_timestamp(symbol, interval)
        # Get previous kline time in miliseconds
        prev_kline_miliseconds = self.market[symbol]["k"]["t"] - kline_miliseconds
        # Limit the query to last max_klines klines
        limit = int(now_miliseconds - max_klines)
        start_time = max(limit, archive_timestamp)
        if start_time >= prev_kline_miliseconds:
            # Query is not needed, set start_time to 0
            start_time = 0
        else:
            # Start from next kline
            start_time += kline_miliseconds
        logger.info(
            f"\n{symbol} - Archive datetime (UTC): "
            f"{miliseconds_to_utc(archive_timestamp)}\n"
            f"{symbol} - Previous candle (UTC): "
            f"{miliseconds_to_utc(prev_kline_miliseconds)}"
        )
        return start_time
