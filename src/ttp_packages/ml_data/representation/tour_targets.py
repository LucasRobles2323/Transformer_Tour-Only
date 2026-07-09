#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/representation/tour_targets.py

"""Conversión de tours a targets supervisados.

Este módulo transforma tours representados como permutaciones en formatos útiles
para entrenamiento, como siguiente ciudad o matriz de adyacencia dirigida.
"""

from __future__ import annotations

from typing import List

import torch

def tour_to_next(tour: List[int]) -> torch.Tensor:
    """Convierte un tour en un vector de siguiente ciudad.

    Args:
        tour: Lista de ciudades visitadas en orden.

    Returns:
        Tensor ``long`` de shape ``(N,)`` donde ``out[i]`` es la ciudad visitada
        inmediatamente después de ``i``.
    """
    tour_tensor = torch.as_tensor(tour, dtype=torch.long)
    nxt = torch.empty_like(tour_tensor)

    # Asigna simultáneamente cada ciudad origen a su siguiente ciudad en el ciclo.
    nxt[tour_tensor] = torch.roll(tour_tensor, shifts=-1)

    return nxt


def tour_to_adj(tour: List[int]) -> torch.Tensor:
    """Convierte un tour en una matriz de adyacencia dirigida.

    Args:
        tour: Lista de ciudades visitadas en orden.

    Returns:
        Matriz ``float32`` de shape ``(N, N)`` con valor ``1.0`` en cada arista
        dirigida del ciclo.
    """
    tour_tensor = torch.as_tensor(tour, dtype=torch.long)
    n_cities = int(tour_tensor.numel())

    adj = torch.zeros((n_cities, n_cities), dtype=torch.float32)

    # Marca todas las aristas del ciclo, incluyendo retorno al inicio.
    adj[tour_tensor, torch.roll(tour_tensor, shifts=-1)] = 1.0

    return adj