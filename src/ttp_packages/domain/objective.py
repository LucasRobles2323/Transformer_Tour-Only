#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/domain/objective.py

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .constants import INF, NEG_INF
from .instance import TTPInstance


def _open_tour(tour: List[int]) -> List[int]:
    """Convierte un tour cerrado explícito en tour abierto.

    Args:
        tour: Secuencia de ciudades. Puede venir como ``[0, ..., 0]``.

    Returns:
        Tour sin la ciudad final repetida si el ciclo venía cerrado.
    """
    if len(tour) >= 2 and int(tour[0]) == int(tour[-1]):
        # El retorno al inicio ya se calcula con np.roll; quitar el último nodo
        # evita agregar un arco artificial 0->0 y recoger dos veces en el depósito.
        return list(map(int, tour[:-1]))

    return list(map(int, tour))


def _item_values(inst: TTPInstance) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construye arrays de profit y peso para los ítems de la instancia.

    Args:
        inst: Instancia TTP con la lista de ítems cargada.

    Returns:
        Tupla ``(profits, weights)`` donde cada array está indexado por el ID
        global del ítem.
    """
    profits = np.fromiter(
        (item.profit for item in inst.items),
        dtype=np.float64,
        count=len(inst.items),
    )
    weights = np.fromiter(
        (item.weight for item in inst.items),
        dtype=np.float64,
        count=len(inst.items),
    )
    return profits, weights


def calculate_speed_coef(inst: TTPInstance) -> float:
    """Calcula la pérdida de velocidad por unidad de peso.

    Args:
        inst: Instancia TTP con capacidad y velocidades definidas.

    Returns:
        Coeficiente lineal de reducción de velocidad. Retorna ``0.0`` cuando
        la capacidad no es positiva.
    """
    if inst.capacity <= 0:
        return 0.0

    return (inst.max_speed - inst.min_speed) / float(inst.capacity)


def calculate_profit(inst: TTPInstance, tour: List[int], packing: List[int]) -> float:
    """Calcula el beneficio total de los ítems seleccionados.

    Args:
        inst: Instancia TTP con ciudades e ítems.
        tour: Secuencia de ciudades visitadas.
        packing: Vector binario donde ``packing[i]`` indica si se recoge el ítem
            con ID global ``i``.

    Returns:
        Beneficio total de los ítems seleccionados en las ciudades del tour.
    """
    if len(tour) == 0 or len(packing) == 0:
        return 0.0

    tour_open = _open_tour(tour)

    profits, _ = _item_values(inst)
    packing_mask = np.asarray(packing, dtype=bool)

    selected_item_ids = [
        item_id
        for city_id in tour_open
        for item_id in inst.cities[int(city_id)].items
    ]

    if not selected_item_ids:
        return 0.0

    item_ids = np.asarray(selected_item_ids, dtype=np.int64)

    # Solo se consideran ítems ubicados en ciudades presentes en el tour.
    return float(profits[item_ids][packing_mask[item_ids]].sum())


def calculate_time(inst: TTPInstance, tour: List[int], packing: List[int]) -> float:
    """Calcula el tiempo total del tour considerando peso acumulado.

    En TTP la velocidad disminuye a medida que la mochila acumula peso. Por eso,
    el tiempo de cada arco depende de todos los ítems recogidos antes de recorrer
    ese arco.

    Args:
        inst: Instancia TTP.
        tour: Secuencia de ciudades visitadas.
        packing: Vector binario de selección de ítems.

    Returns:
        Tiempo total del recorrido. Retorna ``INF`` si la solución excede la
        capacidad o si el tour es demasiado corto.

    Raises:
        RuntimeError: Si no se puede crear la matriz de distancias.
        ValueError: Si la capacidad o la velocidad mínima son inválidas.
    """
    if inst.distance_matrix is None:
        inst.create_distance_matrix()

    if inst.distance_matrix is None:
        raise RuntimeError("No se pudo crear la matriz de distancias.")

    tour_open = _open_tour(tour)

    if len(tour_open) < 2:
        return INF

    if inst.capacity <= 0:
        raise ValueError(f"inst.capacity debe ser mayor que 0. Capacity={inst.capacity}")

    if inst.min_speed <= 0:
        raise ValueError(f"inst.min_speed debe ser mayor que 0. Min speed={inst.min_speed}")

    tour_arr = np.asarray(tour_open, dtype=np.int64)
    packing_mask = np.asarray(packing, dtype=bool)
    dist_matrix = np.asarray(inst.distance_matrix, dtype=np.float64)

    _, item_weights = _item_values(inst)

    city_pick_weights = np.zeros(inst.n_cities, dtype=np.float64)

    for city_id, city in enumerate(inst.cities):
        if not city.items:
            continue

        item_ids = np.asarray(city.items, dtype=np.int64)

        # Cada ciudad puede tener una cantidad distinta de ítems, por eso esta
        # parte se mantiene como bucle por ciudad.
        city_pick_weights[city_id] = item_weights[item_ids][packing_mask[item_ids]].sum()

    cumulative_weights = np.cumsum(city_pick_weights[tour_arr])

    if np.any(cumulative_weights > inst.capacity):
        return INF

    speed_coef = calculate_speed_coef(inst)
    speeds = np.maximum(
        inst.min_speed,
        inst.max_speed - cumulative_weights * speed_coef,
    )

    # np.roll genera el siguiente nodo de cada arco y también cierra el ciclo.
    # Ejemplo: [0, 1, 2] -> [1, 2, 0], calculando también 2->0.
    next_tour_arr = np.roll(tour_arr, -1)
    edge_distances = dist_matrix[tour_arr, next_tour_arr]

    return float(np.sum(edge_distances / speeds))


def calculate_objective_function(
    inst: TTPInstance,
    tour: List[int],
    packing: List[int],
) -> Tuple[float, float, float]:
    """Evalúa una solución completa del problema TTP.

    Args:
        inst: Instancia TTP.
        tour: Secuencia de ciudades visitadas.
        packing: Vector binario de selección de ítems.

    Returns:
        Tupla ``(objective, time, profit)``. Si la solución no es factible,
        retorna ``(NEG_INF, INF, 0.0)``.
    """
    time = calculate_time(inst, tour, packing)

    if time == INF:
        return NEG_INF, INF, 0.0

    profit = calculate_profit(inst, tour, packing)
    objective = profit - (time * inst.rent_per_time)

    return objective, time, profit