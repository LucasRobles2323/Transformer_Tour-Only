#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/packing/initial_packing.py

"""Construcción heurística de packing inicial para un tour fijo.

Este módulo crea una solución inicial TTP-aware para el problema de packing. La
heurística considera profit, densidad profit/peso y una aproximación de impacto
neto sobre el objetivo TTP.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from src.ttp_packages.domain.tour_ops import canonicalize_tour
from src.ttp_packages.optimization.classical.ttp.cs2sa_r.delta_eval import (
    precompute_remaining_distances,
)


def _sample_jitter(rng: random.Random, jitter: float) -> float:
    """Genera ruido de desempate si corresponde.

    Args:
        rng: Generador pseudoaleatorio.
        jitter: Magnitud máxima del ruido uniforme.

    Returns:
        Ruido en ``[-jitter, jitter]`` si ``jitter > 0``; en caso contrario ``0``.
    """
    return rng.uniform(-jitter, jitter) if jitter > 0 else 0.0


def _speed_for_weight(
    *,
    max_speed: float,
    min_speed: float,
    speed_coef: float,
    carried_weight: float,
) -> float:
    """Calcula la velocidad inducida por el peso transportado.

    Args:
        max_speed: Velocidad sin carga.
        min_speed: Velocidad mínima permitida.
        speed_coef: Penalización de velocidad por unidad de peso.
        carried_weight: Peso transportado en la posición actual.

    Returns:
        Velocidad efectiva recortada inferiormente por ``min_speed``.
    """
    speed = max_speed - (carried_weight * speed_coef)
    return speed if speed >= min_speed else min_speed


def build_initial_packing_for_fixed_tour(
    inst: Any,
    tour: list[int],
    *,
    rng: Optional[random.Random] = None,
    jitter: float = 0.0,
) -> list[int]:
    """Construye un packing inicial heurístico para un tour fijo.

    La heurística aplica tres fases:
        1. Prioriza profit bruto.
        2. Prioriza densidad ``profit / weight``.
        3. Prioriza beneficio neto aproximado considerando el aumento de tiempo.

    Args:
        inst: Instancia TTP.
        tour: Tour fijo sobre el cual empaquetar.
        rng: Generador pseudoaleatorio opcional.
        jitter: Ruido pequeño para desempatar candidatos.

    Returns:
        Lista binaria de largo ``M`` donde ``1`` indica ítem seleccionado.
    """
    if rng is None:
        rng = random.Random()

    n_cities = int(inst.n_cities)
    n_items = int(inst.m_items)
    capacity = float(inst.capacity)
    rent_per_time = float(inst.rent_per_time)
    max_speed = float(inst.max_speed)
    min_speed = float(inst.min_speed)

    if n_items <= 0 or capacity <= 0:
        return [0] * n_items

    canonical_tour = canonicalize_tour(tour, n_cities=n_cities, start_city=0)

    items = inst.items
    cities = inst.cities

    packing = [0] * n_items
    total_weight = 0.0

    speed_diff = max_speed - min_speed
    speed_coef = speed_diff / capacity if capacity > 0 else 0.0

    remaining_distances = precompute_remaining_distances(inst, canonical_tour)

    # current_weight_by_position[i] representa el peso transportado al llegar a
    # la posición i del tour. Al tomar un ítem en i, afecta a i y posiciones futuras.
    current_weight_by_position = [0.0] * n_cities

    # Fase 0: profit bruto. Fase 1: densidad. Fase 2: beneficio neto aproximado.
    for phase in range(3):
        for position_index, city_id in enumerate(canonical_tour):
            candidates = [
                item_id
                for item_id in cities[city_id].items
                if packing[item_id] == 0
            ]
            if not candidates:
                continue

            if phase == 0:
                candidates.sort(
                    key=lambda item_id: (
                        float(items[item_id].profit) + _sample_jitter(rng, jitter),
                        -float(items[item_id].weight),
                    ),
                    reverse=True,
                )

            elif phase == 1:

                def density_key(item_id: int) -> float:
                    item_weight = float(items[item_id].weight)
                    item_profit = float(items[item_id].profit)

                    if item_weight <= 0:
                        return float("inf")

                    return (item_profit / item_weight) + _sample_jitter(rng, jitter)

                candidates.sort(key=density_key, reverse=True)

            else:
                current_weight = current_weight_by_position[position_index]
                current_speed = _speed_for_weight(
                    max_speed=max_speed,
                    min_speed=min_speed,
                    speed_coef=speed_coef,
                    carried_weight=current_weight,
                )
                current_inv_speed = 1.0 / current_speed
                remaining_distance = remaining_distances[position_index]

                def net_approx_key(item_id: int) -> float:
                    item = items[item_id]
                    item_profit = float(item.profit)
                    item_weight = float(item.weight)

                    if item_weight <= 0:
                        return float("inf") if item_profit > 0 else float("-inf")

                    new_weight = current_weight + item_weight
                    new_speed = _speed_for_weight(
                        max_speed=max_speed,
                        min_speed=min_speed,
                        speed_coef=speed_coef,
                        carried_weight=new_weight,
                    )

                    # Penaliza el profit por el aumento aproximado de tiempo restante.
                    delta_inv_speed = (1.0 / new_speed) - current_inv_speed
                    net_gain = item_profit - (
                        rent_per_time * remaining_distance * delta_inv_speed
                    )

                    return net_gain + _sample_jitter(rng, jitter)

                candidates.sort(key=net_approx_key, reverse=True)

            for item_id in candidates:
                item = items[item_id]
                item_profit = float(item.profit)
                item_weight = float(item.weight)

                if item_profit <= 0:
                    continue

                if item_weight == 0:
                    packing[item_id] = 1
                    continue

                if item_weight < 0:
                    continue

                if total_weight + item_weight > capacity + 1e-12:
                    continue

                if phase == 2:
                    current_weight = current_weight_by_position[position_index]
                    current_speed = _speed_for_weight(
                        max_speed=max_speed,
                        min_speed=min_speed,
                        speed_coef=speed_coef,
                        carried_weight=current_weight,
                    )

                    new_weight = current_weight + item_weight
                    new_speed = _speed_for_weight(
                        max_speed=max_speed,
                        min_speed=min_speed,
                        speed_coef=speed_coef,
                        carried_weight=new_weight,
                    )

                    delta_inv_speed = (1.0 / new_speed) - (1.0 / current_speed)
                    net_gain = item_profit - (
                        rent_per_time
                        * remaining_distances[position_index]
                        * delta_inv_speed
                    )

                    # En la fase neta solo se acepta el ítem si mejora el objetivo aproximado.
                    if net_gain <= 0:
                        continue

                packing[item_id] = 1
                total_weight += item_weight

                # El peso del ítem afecta desde esta posición hasta el final del tour.
                for future_position in range(position_index, n_cities):
                    current_weight_by_position[future_position] += item_weight

    return packing