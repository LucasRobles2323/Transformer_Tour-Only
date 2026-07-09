#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/torch_io.py

"""Helpers de almacenamiento PyTorch para storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    """Guarda un payload PyTorch usando escritura atómica.

    Args:
        payload: Diccionario a guardar.
        path: Ruta final del archivo ``.pt``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp_path))
    tmp_path.replace(path)


def load_torch_dict(
    path: Path,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Carga un archivo PyTorch y valida que sea un diccionario.

    Args:
        path: Ruta del archivo ``.pt``.
        map_location: Dispositivo destino usado por ``torch.load``.

    Returns:
        Diccionario cargado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el contenido cargado no es un diccionario.
    """
    if not path.exists():
        raise FileNotFoundError(f"Archivo PyTorch no encontrado: {path}")

    payload = torch.load(str(path), map_location=map_location)

    if not isinstance(payload, dict):
        raise ValueError(f"Archivo PyTorch inválido: {path}")

    return payload