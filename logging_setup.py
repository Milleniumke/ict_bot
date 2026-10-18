"""Structured logging to console + rotating file."""
import logging
import logging.handlers


def setup_logging(log_file: str = "bot.log", level: int = logging.INFO):
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=5
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
