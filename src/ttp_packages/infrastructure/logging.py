#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/logging.py

"""Utilidades de logging para el proyecto TTP.

Niveles estándar de logging:
    DEBUG: Nivel 10.
    INFO: Nivel 20.
    WARNING: Nivel 30.
    ERROR: Nivel 40.
    CRITICAL: Nivel 50.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = "%(message)s - %(name)s - %(levelname)s - %(asctime)s"


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configura y devuelve un logger estándar del proyecto.

    Args:
        name: Nombre del logger. Normalmente corresponde a ``__name__``.
        log_file: Ruta opcional de un archivo donde se guardarán los logs.
        level: Nivel mínimo de severidad a registrar.

    Returns:
        Logger configurado con salida a consola y, opcionalmente, archivo.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Evita duplicar mensajes si setup_logger() se llama varias veces para el
    # mismo módulo.
    if not _has_handler(logger, logging.StreamHandler):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)

        # Si el logger ya tenía consola pero no archivo, agregamos solo el
        # FileHandler faltante.
        if not _has_file_handler(logger, log_path):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def _has_handler(logger: logging.Logger, handler_type: type[logging.Handler]) -> bool:
    """Indica si un logger ya tiene un handler de cierto tipo.

    Args:
        logger: Logger a inspeccionar.
        handler_type: Clase de handler buscada.

    Returns:
        ``True`` si el logger ya tiene al menos un handler de ese tipo.
    """
    return any(isinstance(handler, handler_type) for handler in logger.handlers)


def _has_file_handler(logger: logging.Logger, log_path: Path) -> bool:
    """Indica si un logger ya escribe en un archivo específico.

    Args:
        logger: Logger a inspeccionar.
        log_path: Ruta del archivo de log.

    Returns:
        ``True`` si ya existe un ``FileHandler`` apuntando a esa ruta.
    """
    target = log_path.resolve()

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            current = Path(handler.baseFilename).resolve()
            if current == target:
                return True

    return False