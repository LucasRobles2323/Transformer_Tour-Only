#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/tsp/heuristics.py

"""Heurísticas clásicas para construir tours TSP.

Este módulo contiene métodos rápidos para generar tours iniciales. Actualmente
incluye vecino más cercano, usado como baseline o como semilla para mejoras.
"""

from __future__ import annotations

from typing import Any


def nearest_neighbor_tour(inst: Any, start: int = 0) -> list[int]:
    """Genera un tour usando la heurística del vecino más cercano.

    La heurística comienza en ``start`` y, en cada paso, visita la ciudad no
    visitada más cercana a la ciudad actual.

    Args:
        inst: Instancia TTP con ``n_cities`` y ``distance_matrix``.
        start: Ciudad inicial del tour. Si está fuera de rango, se usa ``0``.

    Returns:
        Lista con el orden de visita de las ciudades.
    """
    n_cities = int(inst.n_cities)
    if n_cities == 0:
        return []

    current_city = int(start)
    if not 0 <= current_city < n_cities:
        current_city = 0

    if inst.distance_matrix is None:
        inst.create_distance_matrix()

    distance_matrix = inst.distance_matrix
    if distance_matrix is None:
        raise RuntimeError("No se pudo crear la matriz de distancias.")

    unvisited = set(range(n_cities))
    unvisited.remove(current_city)

    tour = [current_city]

    while unvisited:
        best_next_city = -1
        best_distance = float("inf")
        distances_from_current = distance_matrix[current_city]

        # Se mantiene el loop explícito para conservar la lógica original.
        for candidate_city in unvisited:
            candidate_distance = distances_from_current[candidate_city]

            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_next_city = candidate_city

        tour.append(best_next_city)
        unvisited.remove(best_next_city)
        current_city = best_next_city

    return tour