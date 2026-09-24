"""Tiny logging helper so every service prints in the same format."""

import logging


def get_logger(service_name: str) -> logging.Logger:
    logger = logging.getLogger(service_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(f"[{service_name}] %(asctime)s %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Each logger prints for itself; propagating would print every line
        # again through any parent ("token_daddy") that also has a handler.
        logger.propagate = False
    return logger
