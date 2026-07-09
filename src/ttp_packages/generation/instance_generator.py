#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/generation/instance_generator.py

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

from src.ttp_packages.domain.entities import City, Item
from src.ttp_packages.domain.instance import TTPInstance
from src.ttp_packages.generation.config import (
    AUX_INSTANCE_NAME,
    CAPACITY_DIVISOR,
    COORD_ROUNDING_PRECISION,
    MAX_ATTEMPTS_MULTIPLIER,
    CorrelationType,
    InstanceGeneratorParams,
    RENT_MODE_EXACT,
    RENT_MODE_FAST,
    RENT_TOUR_MODE_NN_2OPT,
)
from src.ttp_packages.generation.math.item_sampling import generate_items_logic
from src.ttp_packages.generation.math.rent_estimation import compute_rent_factor
from src.ttp_packages.infrastructure.logging import setup_logger

# Inicialización del logger para este módulo
logger = setup_logger(__name__)

def generate_ttp_instance(
    params: InstanceGeneratorParams,
    rng: Optional[random.Random] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> TTPInstance:
    """Genera una instancia sintética completa del Traveling Thief Problem.

    La generación crea ciudades con coordenadas únicas, asigna ítems según el
    patrón de correlación configurado, calcula la capacidad de la mochila y
    estima siempre el factor de renta ``R``.

    Args:
        params: Parámetros de generación de la instancia.
        rng: Generador pseudoaleatorio opcional. Si es ``None``, se crea uno.
        log_fn: Función opcional para registrar mensajes. Si es ``None``, se usa
            el logger del módulo.

    Returns:
        Instancia TTP con matriz de distancias precalculada y ``rent_per_time``
        estimado.

    Raises:
        ValueError: Si el número de ciudades, la correlación o el modo de renta
            son inválidos.
        RuntimeError: Si no se logran generar coordenadas únicas o la matriz de
            distancias no puede crearse.
    """
    if log_fn is None:
        log_fn = logger.info

    if rng is None:
        rng = random.Random()

    try:
        corr_type = CorrelationType(params.corr_type_value)
    except ValueError as exc:
        raise ValueError(
            f"'{params.corr_type_value}' no es un CorrelationType válido."
        ) from exc

    if params.n_cities < 3:
        raise ValueError("Se requieren al menos 3 ciudades para formar un tour válido.")

    if params.item_factor < 0:
        raise ValueError("item_factor no puede ser negativo.")

    if params.verbose:
        log_fn(
            f"Iniciando generación TTP: {params.n_cities} ciudades, "
            f"{params.item_factor} ítems/ciudad, corr={corr_type.value}"
        )

    # -----------------------------------------------------------------------
    # 1) Coordenadas únicas
    # -----------------------------------------------------------------------
    min_x, max_x, min_y, max_y = params.coord_range
    used_coords: Set[Tuple[float, float]] = set()
    cities: List[City] = []

    max_attempts = params.n_cities * MAX_ATTEMPTS_MULTIPLIER
    attempts = 0

    for city_id in range(params.n_cities):
        while True:
            x = round(rng.uniform(min_x, max_x), COORD_ROUNDING_PRECISION)
            y = round(rng.uniform(min_y, max_y), COORD_ROUNDING_PRECISION)
            coord = (x, y)

            if coord not in used_coords:
                used_coords.add(coord)
                cities.append(City(city_id, x, y))
                break

            # Este contador solo avanza cuando hay colisión, porque el problema
            # real es quedarse sin coordenadas únicas disponibles.
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "No se pudieron generar coordenadas únicas. "
                    "Aumenta 'coord_range' o reduce 'n_cities'."
                )

    # -----------------------------------------------------------------------
    # 2) Ítems
    # -----------------------------------------------------------------------
    items: List[Item] = []
    global_item_id = 0

    # La ciudad 0 se trata como depósito y actualmente no recibe ítems.
    for city_id in range(1, params.n_cities):
        generated_data = generate_items_logic(
            params.item_factor,
            corr_type,
            rng=rng,
        )

        for profit, weight in generated_data:
            item = Item(global_item_id, city_id, profit, weight)
            items.append(item)
            cities[city_id].items.append(item.id)
            global_item_id += 1

    # -----------------------------------------------------------------------
    # 3) Capacidad
    # -----------------------------------------------------------------------
    total_weight = sum(item.weight for item in items)
    capacity = max(1, int((params.weight_category / CAPACITY_DIVISOR) * total_weight))

    # -----------------------------------------------------------------------
    # 4) Instancia auxiliar y matriz de distancias
    # -----------------------------------------------------------------------
    inst_aux = TTPInstance(
        path=Path(AUX_INSTANCE_NAME),
        name=AUX_INSTANCE_NAME,
        cities=cities,
        items=items,
        n_cities=len(cities),
        m_items=len(items),
        capacity=capacity,
        min_speed=params.min_speed,
        max_speed=params.max_speed,
        rent_per_time=0.0,
    )

    inst_aux.create_distance_matrix()
    dm = inst_aux.distance_matrix

    if dm is None:
        raise RuntimeError("No se pudo crear la matriz de distancias.")

    # -----------------------------------------------------------------------
    # 5) Factor de renta R
    # -----------------------------------------------------------------------
    mode = str(params.compute_R_mode).strip().lower()

    if params.verbose:
        log_fn(
            f"Calculando R | mode={mode} | "
            f"tour_mode={params.compute_R_tour_mode} | eps={params.compute_R_eps}"
        )

    if mode == RENT_MODE_EXACT: 
        # El modo exact usa una configuración fija para evitar depender de heurísticas
        # más rápidas o time-limited configuradas para RENT_MODE_FAST.
        rent_value = compute_rent_factor(
            inst_aux,
            eps=float(params.compute_R_eps),
            tour_mode=str(params.compute_R_tour_mode).strip().lower(),
            two_opt_iters=max(0, int(params.compute_R_2opt_iters)),
            tsp_time_limit_s=float(params.compute_R_tsp_time_limit_s),
            knapsack_mode="auto",
            knapsack_time_limit_s=float(params.compute_R_knapsack_time_limit_s),
            knapsack_gap=float(params.compute_R_knapsack_gap),
            knapsack_hide_output=bool(params.compute_R_knapsack_hide_output),
            verbose=params.verbose,
            log_fn=log_fn,
        )

    elif mode == RENT_MODE_FAST:
        rent_value = compute_rent_factor(
            inst_aux,
            eps=float(params.compute_R_eps),
            tour_mode=str(params.compute_R_tour_mode).strip().lower(),
            two_opt_iters=max(0, int(params.compute_R_2opt_iters)),
            tsp_time_limit_s=float(params.compute_R_tsp_time_limit_s),
            knapsack_mode=str(params.compute_R_knapsack_mode).strip().lower(),
            knapsack_time_limit_s=float(params.compute_R_knapsack_time_limit_s),
            knapsack_gap=float(params.compute_R_knapsack_gap),
            knapsack_hide_output=bool(params.compute_R_knapsack_hide_output),
            verbose=params.verbose,
            log_fn=log_fn,
        )

    else:
        raise ValueError(
            f"compute_R_mode inválido: {params.compute_R_mode}. "
            f"Valores válidos: '{RENT_MODE_EXACT}', '{RENT_MODE_FAST}'."
        )

    if params.verbose:
        log_fn(f"Factor de renta calculado: {rent_value:.6f}")

    # -----------------------------------------------------------------------
    # 6) Ensamble final
    # -----------------------------------------------------------------------
    final_name = (
        f"bench_{params.n_cities}_F{params.item_factor}_"
        f"C{params.weight_category}_{corr_type.value}"
    )

    if params.verbose:
        log_fn(f"Instancia '{final_name}' generada exitosamente.")

    return TTPInstance(
        path=Path(final_name),
        name=final_name,
        cities=cities,
        items=items,
        n_cities=len(cities),
        m_items=len(items),
        capacity=capacity,
        min_speed=params.min_speed,
        max_speed=params.max_speed,
        rent_per_time=rent_value,
        distance_matrix=dm.copy(),
    )