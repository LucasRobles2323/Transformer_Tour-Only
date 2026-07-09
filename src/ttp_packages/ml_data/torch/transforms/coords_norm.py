#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/torch/transforms/coords_norm.py

from __future__ import annotations

try:
    import torch
    from typing import List, Tuple

    from src.ttp_packages.infrastructure.logging import setup_logger
    from src.ttp_packages.ml_data.config import DEFAULT_EPS
except ImportError as e:
    print("\n".join([
        "[ml_data/torch/transforms/coords_norm.py] Error al importar módulos requeridos:",
        str(e),
        "Deteniendo ejecución..."
    ]))
    raise SystemExit(-2)

logger = setup_logger(__name__)


def normalize_coords_minmax(coords_xy: List[Tuple[float, float]], eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Normaliza una lista de coordenadas 2D al rango [0,1] usando Min-Max.

    Esta función procesa una única instancia y aplica la normalización de forma 
    independiente para cada eje (X e Y).

    Args:
        coords_xy (List[Tuple[float, float]]): Lista de tuplas con las coordenadas (X, Y).
        eps (float, optional): Constante pequeña para evitar la división por cero 
            si todos los puntos comparten la misma coordenada. 
            Por defecto es DEFAULT_EPS.

    Returns:
        torch.Tensor: Tensor de forma (N, 2) y tipo float32 con los valores escalados.
    """
    xs = torch.tensor([c[0] for c in coords_xy], dtype=torch.float32)
    ys = torch.tensor([c[1] for c in coords_xy], dtype=torch.float32)
    
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    
    xs = (xs - xmin) / (xmax - xmin + eps)
    ys = (ys - ymin) / (ymax - ymin + eps)
    
    return torch.stack([xs, ys], dim=-1)  # (N,2)


def normalize_coords_minmax_batched(coords_raw: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Normaliza un batch de coordenadas al rango [0,1] usando Min-Max por instancia.

    Diseñada para ejecutarse de manera vectorizada en GPU. La normalización se 
    calcula independientemente para cada instancia en el batch y para cada eje.

    Args:
        coords_raw (torch.Tensor): Tensor original de coordenadas de forma (B, N, 2).
        eps (float, optional): Constante pequeña para evitar división por cero.
            Por defecto es DEFAULT_EPS.

    Returns:
        torch.Tensor: Tensor normalizado de forma (B, N, 2) con valores entre 0 y 1.

    Raises:
        ValueError: Si el tensor de entrada no tiene la forma esperada (B, N, 2).
    """
    if coords_raw.ndim != 3 or coords_raw.shape[-1] != 2:
        error_msg = f"coords_raw debe ser (B,N,2). Recibido {tuple(coords_raw.shape)}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    xs = coords_raw[..., 0]
    ys = coords_raw[..., 1]

    xmin = xs.amin(dim=1, keepdim=True)
    xmax = xs.amax(dim=1, keepdim=True)
    ymin = ys.amin(dim=1, keepdim=True)
    ymax = ys.amax(dim=1, keepdim=True)

    xs = (xs - xmin) / (xmax - xmin + eps)
    ys = (ys - ymin) / (ymax - ymin + eps)
    
    return torch.stack([xs, ys], dim=-1)