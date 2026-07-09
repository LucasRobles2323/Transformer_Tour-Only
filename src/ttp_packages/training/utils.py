#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/training/utils.py

"""Utilidades generales para entrenamiento."""

from __future__ import annotations

import os
import random
from typing import Any

import torch


ENV_PYTHONHASHSEED = "PYTHONHASHSEED"
DEVICE_CUDA = "cuda"


def seed_everything(seed: int) -> None:
    """Fija semillas para mejorar reproducibilidad.

    Args:
        seed: Semilla usada por Python, PyTorch y CUDA.
    """
    random.seed(seed)
    os.environ[ENV_PYTHONHASHSEED] = str(seed)

    torch.manual_seed(seed)

    # No falla en CPU; simplemente deja preparada la semilla para CUDA si aplica.
    torch.cuda.manual_seed_all(seed)


def move_to_device(x: Any, device: torch.device) -> Any:
    """Mueve tensores o estructuras anidadas a un dispositivo.

    Args:
        x: Tensor, diccionario, lista, tupla u otro objeto.
        device: Dispositivo destino.

    Returns:
        La misma estructura con todos los tensores movidos a ``device``.
    """
    if torch.is_tensor(x):
        return x.to(device)

    if isinstance(x, dict):
        # Mantiene la misma estructura del batch, moviendo solo sus tensores.
        return {
            key: move_to_device(value, device)
            for key, value in x.items()
        }

    if isinstance(x, list):
        return [move_to_device(value, device) for value in x]

    if isinstance(x, tuple):
        return tuple(move_to_device(value, device) for value in x)

    # Objetos no tensoriales, como strings o metadata, se dejan intactos.
    return x