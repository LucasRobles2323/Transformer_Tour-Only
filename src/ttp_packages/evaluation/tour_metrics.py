#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/evaluation/tour_metrics.py

"""Métricas auxiliares para comparar tours.

Este módulo calcula distancias, conjuntos de aristas y gaps relativos usados por
la evaluación de modelos y baselines.
"""

from __future__ import annotations

from typing import Any, List, Set, Tuple


def tour_edge_set(tour: List[int]) -> Set[Tuple[int, int]]:
    """Extrae las aristas dirigidas de un tour.

    Incluye la arista de retorno desde la última ciudad hacia la primera.

    Args:
        tour: Secuencia de ciudades del tour.

    Returns:
        Conjunto de aristas dirigidas ``(origen, destino)``.
    """
    n = len(tour)
    return {(tour[i], tour[(i + 1) % n]) for i in range(n)}


def tour_distance(instance: Any, tour: List[int]) -> float:
    """Calcula la distancia total de un tour.

    Args:
        instance: Instancia TTP con matriz de distancias.
        tour: Secuencia de ciudades del tour.

    Returns:
        Distancia total del ciclo, incluyendo retorno a la primera ciudad.

    Raises:
        RuntimeError: Si no se puede crear la matriz de distancias.
    """
    if not tour:
        return 0.0

    if instance.distance_matrix is None:
        instance.create_distance_matrix()

    dm = instance.distance_matrix
    if dm is None:
        raise RuntimeError("No se pudo crear la matriz de distancias.")

    # Import local: NumPy solo se necesita para vectorizar esta métrica.
    import numpy as np

    # Vectoriza el ciclo completo, incluyendo el retorno desde la última ciudad.
    tour_arr = np.asarray(tour, dtype=np.intp)
    next_arr = np.roll(tour_arr, -1)
    dm_arr = np.asarray(dm, dtype=np.float64)

    return float(dm_arr[tour_arr, next_arr].sum())


def compute_relative_gap(value: float, reference: float) -> float:
    """Calcula el gap relativo entre un valor y una referencia.

    En maximización, un valor negativo indica que ``value`` es peor que
    ``reference``.

    Args:
        value: Valor evaluado.
        reference: Valor de referencia.

    Returns:
        Gap relativo estabilizado con un epsilon pequeño.
    """
    return float((value - reference) / (abs(reference) + 1e-9))