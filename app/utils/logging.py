"""Logging utilities."""

from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QObject, Signal


class QtLogEmitter(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, emitter: QtLogEmitter) -> None:
        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._emitter.message.emit(msg)


def configure_logger(name: str = "podflow") -> tuple[logging.Logger, QtLogEmitter]:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger, QtLogEmitter()
    logger.setLevel(logging.INFO)
    emitter = QtLogEmitter()
    handler = QtLogHandler(emitter)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, emitter
