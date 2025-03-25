import logging
from config import LOG_FILE, LOG_LEVEL

def setup_logging():
    """Configure logging for the application."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=LOG_LEVEL,
        format=log_format,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )