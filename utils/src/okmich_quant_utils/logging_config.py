import logging
import os


def setup_logging(folder_path, log_file_name, log_level=logging.INFO):
    logger = logging.getLogger()

    if logger.handlers:
        return logger

    os.makedirs(folder_path, exist_ok=True)

    log_file = os.path.join(folder_path, log_file_name)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
