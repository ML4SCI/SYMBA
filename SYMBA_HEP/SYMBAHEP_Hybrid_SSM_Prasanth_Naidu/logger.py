import logging

def get_logger(name=__name__):
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Prevent adding multiple handlers if logger already has one
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
