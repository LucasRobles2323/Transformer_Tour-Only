#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/representation/instance_solution.py

"""Conversión de instancias y soluciones TTP a samples tensoriales.

Este módulo transforma una instancia TTP y una solución asociada en un sample
compacto con entradas para el modelo y targets de supervisión.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from src.ttp_packages.domain.tour_ops import canonicalize_tour
from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.ml_data.config import (
    KEY_CAPACITY,
    KEY_COORDS_RAW,
    KEY_ITEM_CITY,
    KEY_ITEM_PROFIT,
    KEY_ITEM_WEIGHT,
    KEY_MAX_SPEED,
    KEY_MIN_SPEED,
    KEY_OBJECTIVE,
    KEY_PICKS,
    KEY_PROFIT,
    KEY_RENT,
    KEY_TIME,
    KEY_TOUR_NEXT,
)
from src.ttp_packages.ml_data.representation.tour_targets import tour_to_next

# Configuración del logger para este módulo
logger = setup_logger(__name__)


def instance_solution_to_sample(
    inst: Any,
    sol: Any,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Convierte una instancia y una solución TTP en un sample tensorial.

    El sample usa un formato compacto ``O(N + M)``: almacena coordenadas,
    parámetros globales, atributos de ítems, tour target y picks, pero evita
    matrices cuadráticas como distancias o máscaras.

    Args:
        inst: Instancia TTP con ciudades, ítems y parámetros del problema.
        sol: Solución TTP con tour, packing y métricas.
        verbose: Si es True, registra información de la conversión.
        log_fn: Función opcional de logging.

    Returns:
        Diccionario con secciones ``meta``, ``inputs`` y ``teacher``.
    """
    if log_fn is None:
        log_fn = logger.info

    n_cities = int(inst.n_cities)
    m_items = int(inst.m_items)

    # ----- coords RAW (reales) -----
    coords_xy = [(city.x, city.y) for city in inst.cities]
    coords_raw = torch.tensor(coords_xy, dtype=torch.float32)  # (n_cities,2)

    # ----- params -----
    capacity = float(inst.capacity)
    min_speed = float(inst.min_speed)
    max_speed = float(inst.max_speed)
    rent_per_time = float(inst.rent_per_time)

    # ----- items -----
    item_city = torch.tensor(
        [item.city_id for item in inst.items],
        dtype=torch.long,
    )  # (m_items,)
    item_profit = torch.tensor(
        [item.profit for item in inst.items],
        dtype=torch.float32,
    )  # (m_items,)
    item_weight = torch.tensor(
        [item.weight for item in inst.items],
        dtype=torch.float32,
    )  # (m_items,)

    # ----- teacher tour/picks -----
    tour = canonicalize_tour(tour=sol.tour, n_cities=n_cities)
    tour_next = tour_to_next(tour)  # (n_cities,)

    picks = torch.tensor(list(map(int, sol.packing)), dtype=torch.float32)  # (m_items,)

    # ----- teacher metrics -----
    profit = float(getattr(sol, "profit", 0.0))
    time = float(getattr(sol, "time", 0.0))
    objective = float(getattr(sol, "objective", 0.0))

    sample = {
        "meta": {
            "name": getattr(inst, "name", ""),
            "n_cities": n_cities,
            "m_items": m_items,
            "type_correlation": getattr(inst, "type_correlation", None),
        },
        "inputs": {
            KEY_COORDS_RAW: coords_raw,
            KEY_CAPACITY: torch.tensor([capacity], dtype=torch.float32),
            KEY_ITEM_CITY: item_city,
            KEY_ITEM_PROFIT: item_profit,
            KEY_ITEM_WEIGHT: item_weight,
            KEY_MIN_SPEED: torch.tensor([min_speed], dtype=torch.float32),
            KEY_MAX_SPEED: torch.tensor([max_speed], dtype=torch.float32),
            KEY_RENT: torch.tensor([rent_per_time], dtype=torch.float32),
        },
        "teacher": {
            KEY_TOUR_NEXT: tour_next,
            KEY_PICKS: picks,
            KEY_PROFIT: profit,
            KEY_TIME: time,
            KEY_OBJECTIVE: objective,

            # opcional (no se usa para el .pt compacto)
            "tour": tour,
        },
    }

    if verbose:
        log_fn(
            f"Sample creado exitosamente para {sample['meta']['name']} "
            f"(N={n_cities}, M={m_items})"
        )

    return sample