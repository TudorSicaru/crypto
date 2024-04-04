import asyncio
import configparser
from datetime import datetime, timezone

import asyncmy
import pandas as pd
from databases import Database

# Using the lib_logger
from project_logging import lib_logger as logger

CONFIG_FILE = "config.ini"


class AsyncDB:
    """Wrap the Database class to offer more APIs to interact with MariaDB"""

    def __init__(self, config=None):
        """Sets attributes and loads the config if needed."""
        self.connected = False
        self.config = None
        self.database = None
        self.load_config(config)

    def load_config(self, config):
        """Sets the config information required for initdb method."""
        try:
            if not config:
                config = configparser.ConfigParser()
                config.read(CONFIG_FILE)
            self.config = config["database"]
        except (FileNotFoundError, KeyError) as err:
            logger.error(f"Failed to read DB config, reason: {repr(err)}")

    def initdb(self):
        """Initialize the DB based on the config.ini file."""
        if self.database:
            return
        try:
            database_url = (
                f"{self.config['driver']}://"
                f"{self.config['user']}:"
                f"{self.config['password']}@"
                f"{self.config['address']}/"
                f"{self.config['dbname']}"
            )
            self.database = Database(database_url)
            logger.info(f"Database initialized successfully.")
        except KeyError as err:
            logger.error(f"Initdb failed. Reason: {repr(err)}")

    async def __aenter__(self):
        """__aenter__ method wich allows usage of async with protocol"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """__aexit__ method which allows for usage of async with protocol"""
        await self.disconnect()

    async def connect(self):
        """Check of database is initialized, the connects if it's not connected"""
        if not self.connected:
            self.initdb()
            logger.info(
                "Starting db connection to "
                f"{self.config['user']}@{self.config['address']}"
            )
            await self.database.connect()
            self.conneted = True

    async def disconnect(self):
        """Disconnects the database"""
        await self.database.disconnect()
        logger.info(f"Closed db conn {self.config['user']}@{self.config['address']}")

    async def fetch_one(self, query, values=None):
        """Wraps the fetch_one method of self.database"""
        return await self.database.fetch_one(query=query, values=values)

    async def fetch_all(self, query, values=None):
        """Wraps the fetch_all method of self.database"""
        return await self.database.fetch_all(query=query, values=values)

    async def execute(self, query, values=None):
        """Wraps the execute method of self.database"""
        return await self.database.execute(query=query, values=values)

    async def execute_many(self, query, values=None):
        """Wraps the execute_many method of self.database.
        Used to insert bulk data."""
        return await self.database.execute_many(query=query, values=values)

    def db_to_utc_dt(self, dt_obj):
        """We store db datetimes as NAIVE UTC dates, this method converts them
        to AWARE UTC datetimes.

        NOTE: ALWAYS use this method when you return a datetime object from the
        database, IF you want to make that object timezone-aware.
        """
        if not isinstance(dt_obj, datetime):
            raise ValueError(
                f"db_to_utc_dt: Please provide a datetime object as arg, not: {dt_obj}"
            )
        return dt_obj.replace(tzinfo=timezone.utc)

    async def get_latest_archive_timestamp(self, symbol, interval):
        """Returns the latest timestamp that was archived for the symbol
        with a specific interval.

        Parameters:
            symbol (str): Name of the pair (e.g. BTCUSDT, or ETHUSTD, or ADAUSDT)
            interval (str): Candle interval (e.g. 1m, 1h, 2h, 1d, 1w, 1M)
        """
        query = f"""
            SELECT hd.open_timestamp FROM historical_data hd
            LEFT JOIN pair p ON hd.pair_id=p.id
            WHERE p.symbol="{symbol}" AND hd.interval="{interval}"
            ORDER BY hd.open_timestamp DESC
            LIMIT 1
        """
        logger.debug(f"get_latest_archive_timestamp - Sending `{query}`")
        result = await self.fetch_all(query=query)
        if not result:
            result = 1483228800000
        else:
            result = result[0]["open_timestamp"]
        return result

    async def fetch_archive_data(self, symbol, interval, limit=None):
        """Fetches archive data for the given symbol/interval. All data
        will be fetched if no `limit` is specified."""
        try:
            query = f"""
                SELECT hd.open_price, hd.high, hd.low, hd.close_price, hd.volume, hd.number_of_trades
                FROM historical_data hd
                LEFT JOIN pair p ON hd.pair_id=p.id
                WHERE p.symbol='{symbol}' AND hd.interval='{interval}'
                ORDER BY hd.open_timestamp
                """
            if limit:
                query += f" LIMIT {limit}"
            logger.debug(f"fetch_archive_data - Sending {query}")
            ret_val = await self.fetch_all(query=query)
        except (KeyError, IndexError, asyncmy.errors.ProgrammingError) as err:
            logger.error(err)
            ret_val = []
        return ret_val

    async def archive_historical_data(self, symbol, interval, klines):
        """Inserts new data to the archive table given the symbol,
        interval and the klines - list of lists"""
        try:
            pair_id = await self.get_pair_id_of_symbol(symbol)
            values = self.prepare_historical_data(pair_id, interval, klines)
            query = """
                INSERT INTO historical_data (pair_id, `interval`, open_timestamp, open_time, open_price, high, low, close_price, volume, number_of_trades)
                VALUES (:pair_id, :interval, :open_timestamp, :open_time, :open_price, :high, :low, :close_price, :volume, :number_of_trades)
            """
            logger.debug(
                f"archive_historical_data({symbol}, {pair_id}) - Sending "
                f"{query}\nValues: {values}"
            )
            await self.execute_many(query, values=values)
            logger.debug(
                f"archive_historical_data({symbol}, {pair_id}) - inserted {len(values)} values"
            )
        except (KeyError, IndexError, asyncmy.errors.ProgrammingError) as err:
            logger.error(err)
            return await self.get_latest_archive_record(
                symbol, interval, date_format="ms"
            )

    def prepare_historical_data(self, pair_id, interval, klines):
        """Creates the query string for the `bulk` insert into historical_data table"""
        # Header of the klines lists
        header = [
            "open_timestamp",
            "open_price",
            "high",
            "low",
            "close_price",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_vol",
            "taker_buy_quote_vol",
            "ignore",
        ]
        # Drop columns we are not interested in
        archive = pd.DataFrame(klines, columns=header)
        archive = archive.drop(
            columns=[
                "close_time",
                "quote_asset_volume",
                "taker_buy_base_vol",
                "taker_buy_quote_vol",
                "ignore",
            ]
        )
        # add pair_id and interval columns
        archive.insert(0, "pair_id", pair_id)
        archive.insert(1, "interval", interval)
        archive["open_time"] = pd.to_datetime(archive["open_timestamp"], unit="ms")
        return archive.to_dict(orient="records")

    async def get_pair_id_of_symbol(self, symbol):
        """Retrieve the id of the pair (symbol) from the `pair` table"""
        query = "SELECT id FROM pair WHERE symbol=:symbol ORDER BY id DESC"
        logger.debug(
            f"get_pair_id_of_symbol - Sending `{query}` with symbol=`{symbol}`"
        )
        pair_id = await self.fetch_one(query=query, values={"symbol": symbol})
        return pair_id["id"]

    async def get_open_trades(self, **filters):
        """Retrieves a list of trades that are still ongoing.

        Parameters:
            filters (dict): They will be key:value pairs
        """
        query = """
            SELECT p.symbol, t.* FROM trade t 
            LEFT JOIN pair p ON t.pair_id=p.id 
            """
        query += self._build_filters_str(**filters)
        logger.debug(f"get_open_trades - Sending `{query}` FILTERS: {filters}")
        return await self.fetch_all(query=query, values=filters)

    async def insert_trade(self, **values):
        """Inserts a new trade record in the 'trade' table.

        Params:
            pair_id (int): REQUIRED. ID of the pair from `pair` table
            open_price (float): REQUIRED. Price at which we opened the trade
            created_at (datetime): OPTIONAL. Automatically set to datetime.now()
            current_price (float): OPTIONAL. Current price of the pair
            close_price (float): OPTIONAL, default 0. When not set, trade_open will
                be equal to 1, to signal that the trade is ongoing
            prediction (float): OPTIONAL. Price prediction returned by the LSTM model
            probability (float): OPTIONAL. The probability of the prediction from LSTM
            stop_loss (float): OPTIONAL. Price below which the trade is closed
            take_profit (float): OPTIONAL. Price above which the trade is closed
            trade_open (int): OPTIONAL, default 0. This flag signals that the trade is
                open or closed. The value is derived from close_price, so we don't have
                to set this column ourselves. It should be treated as `read-only`.
        """
        fields = self._build_fields(**values)
        values_str = self._build_values_str(**values)
        query = f"""
            INSERT INTO trade {fields}
            VALUES {values_str}
            """
        logger.debug(f"insert_trade - Sending `{query}`\n`{values}`")
        return await self.execute(query, values=values)

    async def update_trade(self, trade_id, **values):
        """Updates an existing trade record with specified values.

        Params:
            trade_id (int): REQUIRED. The unique identifier for the trade to update.
            **values: Key-value pairs of the fields to update.
        """
        set_clause = self._build_update_clause(**values)
        query = f"UPDATE trade SET {set_clause} WHERE id = :id"
        values["id"] = trade_id
        logger.debug(f"update_trade - Sending `{query}`\nWith values: {values}")
        return await self.execute(query, values=values)

    def _build_filters_str(self, **filters):
        """Builds the filters for a SELECT statement."""
        filters_str = " AND ".join([f"{key} = :{key}" for key in filters.keys()])
        if filters_str:
            filters_str = " WHERE " + filters_str
        return filters_str

    def _build_update_clause(self, **values):
        """Builds the CLAUSE string of an UPDATE statement."""
        return ", ".join([f"{key} = :{key}" for key in values.keys()])

    def _build_fields(self, **values):
        """Builds a string of fields used during INSERT
        statements, from a given values dictionary.
        """
        return "(" + ", ".join(values) + ")"

    def _build_values_str(self, **values):
        """Builds the values_str to be passed to an INSERT statement."""
        return "(" + ", ".join(f":{key}" for key in values) + ")"
