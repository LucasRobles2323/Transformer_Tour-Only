#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/modeling/utils/features.py

"""Utilidades de transformación y normalización de features.

Este módulo contiene helpers tensoriales usados por el modelo para normalizar
features con máscaras, preparar escalares para broadcasting y aplicar
transformaciones simples de rango.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.ttp_packages.modeling.config import DEFAULT_EPSILON, DEFAULT_MAX_LOG


def masked_standardize(
    x: Tensor,
    mask: Tensor,
    dim: int,
    eps: float = DEFAULT_EPSILON,
) -> Tensor:
    """Estandariza un tensor ignorando posiciones enmascaradas.

    Calcula media y desviación estándar solo sobre posiciones válidas según
    ``mask``. Las posiciones de padding no afectan las estadísticas.

    Args:
        x: Tensor de entrada a estandarizar.
        mask: Máscara binaria con ``1`` para elementos válidos y ``0`` para
            padding.
        dim: Dimensión sobre la cual calcular media y desviación estándar.
        eps: Valor pequeño para evitar divisiones por cero.

    Returns:
        Tensor estandarizado con la misma shape que ``x``.
    """
    values = x.float()
    valid_mask = mask.float()

    valid_count = valid_mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
    mean = (values * valid_mask).sum(dim=dim, keepdim=True) / valid_count

    centered = values - mean
    variance = (centered * valid_mask).pow(2).sum(dim=dim, keepdim=True) / valid_count
    std = variance.sqrt().clamp(min=eps)

    return centered / (std + eps)


def as_b1(
    x: Tensor | float | int,
    B: int,
    device: torch.device,
) -> Tensor:
    """Convierte un escalar o tensor 1D a shape ``(B, 1)``.

    Args:
        x: Valor escalar, tensor escalar o tensor 1D.
        B: Tamaño de batch esperado.
        device: Dispositivo destino.

    Returns:
        Tensor ``float32`` con shape ``(B, 1)`` o compatible.
    """
    batch_size = int(B)

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)

    values = x.float().to(device)

    if values.dim() == 0:
        return values.view(1, 1).expand(batch_size, 1)

    if values.dim() == 1:
        return values.view(-1, 1)

    return values


def log_feat01(x: Tensor, max_log: float = DEFAULT_MAX_LOG) -> Tensor:
    """Aplica transformación logarítmica normalizada.

    Args:
        x: Tensor de entrada. Los valores menores a ``1.0`` se recortan a
            ``1.0`` antes del logaritmo.
        max_log: Valor usado para escalar el logaritmo.

    Returns:
        Tensor transformado por ``log(x) / max_log``.
    """
    return torch.log(x.clamp(min=1.0)) / max_log


def to_pm1(x: Tensor) -> Tensor:
    """Mapea valores desde ``[0, 1]`` hacia ``[-1, 1]``.

    Args:
        x: Tensor con valores esperados en rango ``[0, 1]``.

    Returns:
        Tensor escalado al rango ``[-1, 1]``.
    """
    return x * 2.0 - 1.0