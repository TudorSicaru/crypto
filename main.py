import asyncio
import configparser

from asyncdb import AsyncDB
from market_monitor import MarketMonitor, async_binance_client
from market_oracle import MarketOracle
from project_logging import logger
from trade_manager import TradeManager

CONFIG_FILE = "config.ini"


async def main():
    logger.info(f"Script initialized, reading config from `{CONFIG_FILE}`")
    # Parse the config file
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    # Dictionary to store live market data
    live_market = {}
    # Dictionary to store predictions
    predictions = {}

    async with AsyncDB(config) as database, async_binance_client(config) as client:
        # Initialize the market monitor
        market_monitor = MarketMonitor(client, database, config, live_market)
        # Load pre-trained LSTM model
        oracle = MarketOracle(database, config, predictions)
        # Initialize the trade manager
        trade_manager = TradeManager(client, database, config, live_market, predictions)
        # Setup a list of tasks to execute concurrently
        tasks = [
            asyncio.create_task(market_monitor.track_klines()),
            asyncio.create_task(oracle.predict_pairs()),
            asyncio.create_task(trade_manager.trade_pairs()),
        ]
        logger.info("Running all tasks concurrently. Press CTRL + C to cancel.")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.warning("CTRL + C detected, exiting...")


if __name__ == "__main__":
    asyncio.run(main())
