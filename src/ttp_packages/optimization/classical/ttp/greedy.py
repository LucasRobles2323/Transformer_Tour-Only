#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/ttp/greedy.py

"""Heurísticas voraces simples para construir soluciones TTP.

Este módulo combina un tour por vecino más cercano con un packing voraz basado
en densidad ``profit / weight``. Sirve como baseline rápido y determinista para
comparar contra solvers más costosos.
"""

from __future__ import annotations

from typing import Any

from src.ttp_packages.domain.solution import TTPSolution
from src.ttp_packages.optimization.classical.tsp.heuristics import nearest_neighbor_tour


FLOAT_TOLERANCE = 1e-12
DEFAULT_START_CITY = 0


def solve_tour_nearest_neighbor(inst: Any) -> list[int]:
    """Genera un tour usando la heurística del vecino más cercano.

    Args:
        inst: Instancia TTP con ciudades y matriz de distancias.

    Returns:
        Tour como lista de identificadores de ciudades, rotado para comenzar en
        la ciudad ``0`` cuando esa ciudad está presente.
    """
    tour = nearest_neighbor_tour(inst)

    if tour and tour[0] != DEFAULT_START_CITY and DEFAULT_START_CITY in tour:
        start_position = tour.index(DEFAULT_START_CITY)
        tour = tour[start_position:] + tour[:start_position]

    return tour


def solve_packing_density(inst: Any) -> list[int]:
    """Construye un packing voraz por densidad de profit.

    Ordena los ítems por densidad ``profit / weight`` y selecciona candidatos
    mientras no se exceda la capacidad de la mochila. Los ítems con profit no
    positivo se ignoran.

    Args:
        inst: Instancia TTP con ítems y capacidad de mochila.

    Returns:
        Lista binaria de longitud ``m_items`` donde ``1`` indica ítem
        seleccionado y ``0`` indica ítem ignorado.
    """
    items = inst.items
    n_items = int(inst.m_items)
    capacity = float(inst.capacity)

    packing = [0] * n_items
    if n_items == 0 or capacity <= 0:
        return packing

    def density_key(item_id: int) -> tuple[float, float, float]:
        """Calcula la clave de ordenamiento greedy para un ítem."""
        item = items[item_id]
        profit = float(item.profit)
        weight = float(item.weight)

        if profit <= 0:
            density = float("-inf")
        elif weight <= 0:
            density = float("inf")
        else:
            density = profit / weight

        # Desempate: mayor profit y, luego, menor peso.
        return density, profit, -weight

    item_order = list(range(n_items))
    item_order.sort(key=density_key, reverse=True)

    total_weight = 0.0

    for item_id in item_order:
        item = items[item_id]
        profit = float(item.profit)
        weight = float(item.weight)

        if profit <= 0:
            continue

        # Un peso no positivo no consume capacidad, pero mantiene el profit positivo.
        if weight <= 0:
            packing[item_id] = 1
            continue

        if total_weight + weight <= capacity + FLOAT_TOLERANCE:
            packing[item_id] = 1
            total_weight += weight

    return packing


def solve_greedy_solution(inst: Any) -> TTPSolution:
    """Construye una solución TTP completa con heurísticas voraces.

    Args:
        inst: Instancia TTP a resolver.

    Returns:
        Solución TTP formada por un tour vecino-más-cercano y un packing por
        densidad.
    """
    tour = solve_tour_nearest_neighbor(inst)
    packing = solve_packing_density(inst)

    return TTPSolution(inst, tour, packing)