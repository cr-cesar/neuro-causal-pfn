"""Registro sencillo y reutilizable para entrenamiento y evaluacion."""
import logging


def get_logger(name: str = "neurocausalpfn", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                                datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
