#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/runtime.py

"""Utilidades para seleccionar el dispositivo de ejecución de PyTorch."""

from __future__ import annotations

import torch


DEFAULT_PREFER_CUDA = True


def get_default_device(prefer_cuda: bool = DEFAULT_PREFER_CUDA) -> torch.device:
    """Devuelve el dispositivo PyTorch recomendado.

    Args:
        prefer_cuda: Si es True, usa CUDA cuando esté disponible.

    Returns:
        ``torch.device("cuda:0")`` si CUDA está disponible y se prefiere GPU;
        en caso contrario, ``torch.device("cpu")``.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def get_default_map_location(prefer_cuda: bool = DEFAULT_PREFER_CUDA) -> str:
    """Devuelve el dispositivo por defecto como string para ``torch.load``.

    Args:
        prefer_cuda: Si es True, permite cargar en CUDA cuando esté disponible.

    Returns:
        Representación string del dispositivo, por ejemplo ``"cpu"`` o
        ``"cuda:0"``.
    """
    return str(get_default_device(prefer_cuda=prefer_cuda))