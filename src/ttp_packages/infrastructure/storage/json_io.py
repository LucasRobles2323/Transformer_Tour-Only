#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/json_io.py

"""Helpers de lectura y escritura JSON para storage."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

def _torch_to_jsonable(value: Any) -> tuple[bool, Any]:
    """Convierte valores PyTorch a JSON si corresponde.

    Args:
        value: Valor a convertir.

    Returns:
        Tupla ``(handled, converted_value)``.
    """
    try:
        import torch
    except ImportError:
        return False, None

    if isinstance(value, torch.device):
        return True, str(value)

    if torch.is_tensor(value):
        converted = value.item() if value.dim() == 0 else value.detach().cpu().tolist()
        return True, converted

    return False, None

def to_jsonable(value: Any) -> Any:
    """Convierte un objeto Python en una estructura compatible con JSON.

    Args:
        value: Objeto a convertir.

    Returns:
        Objeto compuesto por tipos serializables por ``json.dump``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}

    if isinstance(value, Enum):
        return value.value

    handled, converted = _torch_to_jsonable(value)
    if handled: 
        return converted

    if isinstance(value, Mapping):
        return {str(key): to_jsonable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    return str(value)


def load_json(path: Path) -> Any:
    """Carga un archivo JSON desde disco.

    Args:
        path: Ruta del archivo JSON.

    Returns:
        Objeto Python deserializado desde el archivo.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, obj: Any) -> None:
    """Guarda un objeto Python como JSON usando escritura atómica.

    Args:
        path: Ruta destino.
        obj: Objeto a serializar.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(obj), file, ensure_ascii=False, indent=2)

    tmp_path.replace(path)