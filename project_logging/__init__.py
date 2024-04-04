"""Sets up logging and initialize the main project logger"""

import logging

from project_logging.logging_config import setup_logging

# Setup logging according to logging_config.yml
setup_logging()

# Initialize a logger called 'main'
logger = logging.getLogger("main")

# Initialize a logger for libraries
lib_logger = logging.getLogger("lib")

# Initialize the lstm logger
lstm_logger = logging.getLogger("lstm")

# Initialize a logger used when training the models
train_logger = logging.getLogger("train_logger")
