#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/ttp/random_solver.py

"""Solvers random para construir soluciones TTP de referencia.

Este módulo implementa dos baselines aleatorios:

    1. ``random``: genera un tour random válido y un packing random factible.
    2. ``rand+krp``: genera un tour random válido y optimiza el packing sobre
       ese tour usando la misma rutina KRP empleada por otros baselines.

Su objetivo no es competir como solver principal, sino entregar referencias
simples para comparar contra modelo, TSP+KRP y CS2SA-R.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, Optional

from src.ttp_packages.domain.solution import TTPSolution


DEFAULT_START_CITY = 0
DEFAULT_PICK_PROBABILITY = 0.5
FLOAT_TOLERANCE = 1e-12


def solve_random_tour(
    inst: Any,
    *,
    seed: Optional[int] = None,
    start_city: int = DEFAULT_START_CITY,
) -> list[int]:
    """Genera un tour random válido.

    El tour visita cada ciudad exactamente una vez y queda rotado para comenzar
    en ``start_city``. No se agrega la ciudad inicial al final, porque el resto
    del proyecto trabaja con tours abiertos y cierra el ciclo al evaluar.

    Args:
        inst: Instancia TTP con ``n_cities``.
        seed: Semilla opcional para reproducibilidad.
        start_city: Ciudad inicial del tour.

    Returns:
        Lista de ciudades que representa una permutación random válida.

    Raises:
        ValueError: Si la instancia no tiene ciudades o ``start_city`` no existe.
    """
    n_cities = int(inst.n_cities)

    if n_cities <= 0:
        raise ValueError("La instancia debe tener al menos una ciudad.")

    if start_city < 0 or start_city >= n_cities:
        raise ValueError(
            f"start_city inválida: {start_city}. "
            f"Debe estar en el rango [0, {n_cities - 1}]."
        )

    rng = random.Random(seed)

    remaining_cities = [
        city_id
        for city_id in range(n_cities)
        if city_id != start_city
    ]
    rng.shuffle(remaining_cities)

    return [start_city] + remaining_cities


def solve_random_packing(
    inst: Any,
    *,
    seed: Optional[int] = None,
    pick_probability: float = DEFAULT_PICK_PROBABILITY,
) -> list[int]:
    """Genera un packing random factible por capacidad.

    Recorre los ítems en orden aleatorio. Para cada ítem decide aleatoriamente
    si intenta tomarlo y solo lo agrega si no supera la capacidad.

    Args:
        inst: Instancia TTP con ``items``, ``m_items`` y ``capacity``.
        seed: Semilla opcional para reproducibilidad.
        pick_probability: Probabilidad de intentar tomar cada ítem.

    Returns:
        Lista binaria de largo ``m_items`` donde ``1`` indica ítem seleccionado.
    """
    n_items = int(inst.m_items)
    capacity = float(inst.capacity)

    packing = [0] * n_items
    if n_items <= 0 or capacity <= 0:
        return packing

    rng = random.Random(seed)
    item_order = list(range(n_items))
    rng.shuffle(item_order)

    total_weight = 0.0
    probability = min(1.0, max(0.0, float(pick_probability)))

    for item_id in item_order:
        # La decisión de tomar o no tomar el ítem es completamente random.
        if rng.random() > probability:
            continue

        item = inst.items[item_id]
        item_weight = float(item.weight)

        # Pesos negativos no son válidos para la mochila; se ignoran
        # defensivamente aunque las instancias TTP normales no deberían tenerlos.
        if item_weight < 0:
            continue

        if total_weight + item_weight <= capacity + FLOAT_TOLERANCE:
            packing[item_id] = 1
            total_weight += item_weight

    return packing


def solve_random_solution(
    inst: Any,
    *,
    seed: Optional[int] = None,
    start_city: int = DEFAULT_START_CITY,
    pick_probability: float = DEFAULT_PICK_PROBABILITY,
) -> TTPSolution:
    """Construye una solución TTP completamente random.

    Usa semillas derivadas para que el tour y el packing sean reproducibles, pero
    independientes entre sí.

    Args:
        inst: Instancia TTP a resolver.
        seed: Semilla opcional para reproducibilidad.
        start_city: Ciudad inicial del tour.
        pick_probability: Probabilidad de intentar tomar cada ítem.

    Returns:
        Solución TTP random con objetivo, tiempo y profit ya calculados.
    """
    base_seed = 0 if seed is None else int(seed)

    tour = solve_random_tour(
        inst,
        seed=base_seed + 101,
        start_city=start_city,
    )
    packing = solve_random_packing(
        inst,
        seed=base_seed + 202,
        pick_probability=pick_probability,
    )

    return TTPSolution(inst, tour, packing)


def solve_random_tour_with_krp(
    inst: Any,
    *,
    time_budget_s: float,
    n_restarts: int,
    seed: Optional[int] = None,
    start_city: int = DEFAULT_START_CITY,
    log_fn: Optional[Callable[[str], None]] = None,
) -> TTPSolution:
    """Construye una solución TTP con tour random y packing KRP.

    Primero genera un tour random válido. Luego mantiene ese tour fijo y llama a
    la rutina clásica de packing para optimizar la selección de ítems. Esto crea
    un baseline comparable al modelo neuronal, porque ambos generan primero un
    tour y después optimizan packing sobre ese tour.

    Args:
        inst: Instancia TTP a resolver.
        time_budget_s: Presupuesto de tiempo para optimizar packing.
        n_restarts: Cantidad de reinicios para la optimización de packing.
        seed: Semilla opcional para reproducibilidad.
        start_city: Ciudad inicial del tour random.
        log_fn: Función opcional de logging.

    Returns:
        Solución TTP con tour random y packing optimizado por KRP.
    """
    # Import local para evitar cargar el solver de packing cuando solo se usa el
    # baseline completamente random.
    from src.ttp_packages.optimization.classical.packing.api import (
        solve_pack_for_fixed_tour,
    )

    base_seed = 0 if seed is None else int(seed)

    # Usa la misma derivación de semilla que solve_random_solution para que,
    # con igual seed, random y rand+krp compartan exactamente el mismo tour.
    tour = solve_random_tour(
        inst,
        seed=base_seed + 101,
        start_city=start_city,
    )

    return solve_pack_for_fixed_tour(
        inst,
        tour,
        time_budget_s=time_budget_s,
        n_restarts=n_restarts,
        seed=base_seed + 303,
        log_fn=log_fn,
    )