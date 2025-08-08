
import logging
import os
from logging.handlers import RotatingFileHandler

# --- Configuration ---
LOG_DIR = "logs"
LOG_FILE = "app.log"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

def setup_logger():
    """Configures and returns a centralized logger."""
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Create a logger
    logger = logging.getLogger("GestureClassifierLogger")
    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all logs

    # Prevent handlers from being added multiple times
    if logger.hasHandlers():
        return logger

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
    )

    # Create a rotating file handler
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create a console handler for printing to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only show INFO and above in the console
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger setup complete.")
    return logger

# Create a single logger instance to be used by the application
logger = setup_logger()
