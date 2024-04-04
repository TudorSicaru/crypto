-- Create historical_data table to archive kline records
CREATE TABLE `historical_data` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `pair_id` int(10) unsigned NOT NULL,
  `interval` char(50) DEFAULT NULL,
  `open_timestamp` bigint(20) unsigned NOT NULL,
  `open_time` datetime NOT NULL,
  `open_price` decimal(19,12) unsigned NOT NULL,
  `high` decimal(19,12) unsigned NOT NULL,
  `low` decimal(19,12) unsigned NOT NULL,
  `close_price` decimal(19,12) unsigned NOT NULL,
  `volume` decimal(25,12) unsigned NOT NULL,
  `number_of_trades` bigint(20) unsigned NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  KEY `pair_id` (`pair_id`),
  CONSTRAINT `symbol_id` FOREIGN KEY (`pair_id`) REFERENCES `pair` (`id`) ON DELETE CASCADE ON UPDATE NO ACTION
) ENGINE=InnoDB AUTO_INCREMENT=40868 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='Table that holds candlestick information for different symbols.';

-- Create the trade table to hold records regarding all open/closed trades
CREATE TABLE `trade` (
  `id` int(20) unsigned NOT NULL AUTO_INCREMENT,
  `pair_id` int(10) unsigned NOT NULL,
  `open_price` decimal(19,12) unsigned NOT NULL,
  `created_at` datetime DEFAULT current_timestamp(),
  `current_price` decimal(19,12) unsigned DEFAULT NULL,
  `close_price` decimal(19,12) unsigned DEFAULT 0.000000000000,
  `prediction` decimal(19,12) unsigned DEFAULT NULL,
  `probability` decimal(13,12) unsigned DEFAULT NULL,
  `stop_loss` decimal(19,12) unsigned DEFAULT NULL,
  `take_profit` decimal(19,12) unsigned DEFAULT NULL,
  `trade_open` tinyint(4) GENERATED ALWAYS AS (if(`close_price` = 0,1,0)) STORED,
  PRIMARY KEY (`id`),
  KEY `idx_trade_open` (`trade_open`),
  KEY `coin_id` (`pair_id`) USING BTREE,
  CONSTRAINT `pair_id` FOREIGN KEY (`pair_id`) REFERENCES `pair` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=64 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='This table holds information regarding the trades made by the bot.';

-- Create the pair table that holds a list of pairs
CREATE TABLE `pair` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `symbol` char(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='Table that holds info regarding which crypto coins we will trade.';

-- Populate a few pairs in the pair table (we need this to be able to trade them)
INSERT INTO `pair` (symbol)
VALUES ('BTCUSDT'), ('ETHUSDT'), ('WIFUSDT');