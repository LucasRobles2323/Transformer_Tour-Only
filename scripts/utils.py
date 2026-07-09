#!/usr/bin/python
# -*- coding: utf-8 -*-

# scripts/utils.py

"""Utilidades compartidas por scripts CLI.

Este módulo contiene helpers pequeños para leer configuración JSON y resolver la
función de logging indicada por los archivos de configuración.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.ttp_packages.application.config import setup_logger


logger = setup_logger(__name__)


def _load_json(path: str | Path) -> Dict[str, Any]:
    """Carga un archivo JSON como diccionario.

    Args:
        path: Ruta al archivo JSON.

    Returns:
        Contenido del archivo como diccionario.
    """
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_log_fn(name: Any) -> Optional[Callable[[str], None]]:
    """Resuelve el modo de logging definido en configuración.

    Args:
        name: Valor leído desde ``log_fn``. Puede ser ``None``, ``"print"``,
            ``"logger"`` o ``"none"``.

    Returns:
        Función de logging resuelta, o ``None`` si se desactiva logging.

    Raises:
        ValueError: Si ``name`` no es un modo de logging soportado.
    """
    if name is None:
        return print

    if not isinstance(name, str):
        raise ValueError("log_fn debe ser null, 'print', 'logger' o 'none'.")

    value = name.strip().lower()

    if value == "print":
        return print

    if value == "logger":
        return logger.info

    if value == "none":
        return None

    raise ValueError("log_fn debe ser 'print', 'logger' o 'none'.")