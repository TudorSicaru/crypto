"""Implements a method to obtain 'standardized' loggers for the app"""

import colorama
import logging.config
import yaml
import os


def setup_logging(default_level=logging.INFO, env_key="LOG_CFG"):
    """Setup logging configuration from a YAML file."""
    # Allow for colored logging
    colorama.init()
    # Load the config from the yml file
    path = os.path.join(os.path.dirname(__file__), "logging_config.yml")
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        # Configure the logging module to use our custom config
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        print(f"Failed to load configuration file: {path}")
