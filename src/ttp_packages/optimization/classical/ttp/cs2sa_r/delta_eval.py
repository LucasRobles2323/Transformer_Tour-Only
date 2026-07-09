#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/ttp/cs2sa_r/delta_eval.py

"""Evaluación completa e incremental de soluciones TTP.

Este módulo calcula pesos, tiempos, profit y objetivo de una solución TTP.
También implementa actualizaciones rápidas para movimientos locales usados por
CS2SA-R, como 2-opt sobre el tour y bit-flips sobre el packing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .route_cache import RouteCache


def _distance_matrix(inst: Any) -> NDArray[np.float64]:
    """Obtiene la matriz de distancias como ``np.ndarray``.

    Si la matriz aún no existe, la crea usando ``inst.create_distance_matrix()``.
    Además, normaliza el resultado a ``np.ndarray`` con dtype ``float64`` para
    que el resto del módulo pueda usar indexación NumPy ``distance_matrix[i, j]`` 
    de forma consistente.

    Args:
        inst: Instancia TTP con atributo ``distance_matrix`` y método
            ``create_distance_matrix()``.

    Returns:
        Matriz de distancias con shape ``(n_cities, n_cities)``.

    Raises:
        RuntimeError: Si la matriz no puede generarse.
        ValueError: Si la matriz generada no tiene shape cuadrado compatible
            con ``inst.n_cities``.
    """
    if inst.distance_matrix is None:
        inst.create_distance_matrix()

    distance_matrix = inst.distance_matrix
    if distance_matrix is None:
        raise RuntimeError("No se pudo crear la matriz de distancias.")

    distance_matrix_arr = np.asarray(distance_matrix, dtype=np.float64)

    if distance_matrix_arr.ndim != 2:
        raise ValueError(
            f"distance_matrix debe ser 2D. Forma recibida: {distance_matrix_arr.shape}."
        )

    if distance_matrix_arr.shape[0] != distance_matrix_arr.shape[1]:
        raise ValueError(
            "distance_matrix debe ser cuadrada. "
            f"Forma recibida: {distance_matrix_arr.shape}."
        )

    n_cities = int(inst.n_cities)
    if distance_matrix_arr.shape != (n_cities, n_cities):
        raise ValueError(
            f"distance_matrix tiene forma {distance_matrix_arr.shape}, "
            f"pero se esperaba {(n_cities, n_cities)}."
        )

    return distance_matrix_arr


def _speed(inst: Any, curr_weight: float) -> float:
    """Calcula la velocidad efectiva según el peso acumulado.

    Args:
        inst: Instancia TTP.
        curr_weight: Peso actual acumulado en la mochila.

    Returns:
        Velocidad actual, acotada inferiormente por ``inst.min_speed``.
    """
    speed_max = float(inst.max_speed)
    speed_min = float(inst.min_speed)
    capacity = float(inst.capacity)

    if capacity <= 0:
        return speed_max

    speed_coef = (speed_max - speed_min) / capacity
    speed_curr = speed_max - (curr_weight * speed_coef)

    return max(speed_min, speed_curr)


def _dist(inst: Any, i: int, j: int) -> float:
    """Obtiene la distancia entre dos ciudades.

    Args:
        inst: Instancia TTP.
        i: Ciudad de origen.
        j: Ciudad de destino.

    Returns:
        Distancia entre ``i`` y ``j``.
    """
    distance_matrix = _distance_matrix(inst)
    return float(distance_matrix[i, j])


def city_items_taken_weight(
    inst: Any,
    packing: list[int],
    city_id: int,
) -> float:
    """Calcula el peso total recogido en una ciudad.

    Args:
        inst: Instancia TTP.
        packing: Lista binaria que representa el plan de recolección.
        city_id: Ciudad evaluada.

    Returns:
        Peso total de los ítems seleccionados en ``city_id``.
    """
    weight = 0.0
    items = inst.items

    for item_id in inst.cities[city_id].items:
        if packing[item_id]:
            weight += items[item_id].weight

    return weight


def recompute_history_full(
    inst: Any,
    tour: list[int],
    packing: list[int],
) -> RouteCache:
    """Recalcula toda la caché física y objetiva de una solución.

    Este cálculo completo se usa para inicializar la caché o para reconstruirla
    después de movimientos estructurales del tour. Calcula pesos acumulados,
    tiempos por arco, tiempos acumulados, profit, peso final, tiempo final y
    objetivo TTP.

    Args:
        inst: Instancia TTP.
        tour: Lista con el orden de visita de las ciudades.
        packing: Lista binaria con los ítems seleccionados.

    Returns:
        Caché completa de la solución.
    """
    n_positions = len(tour)
    if n_positions == 0:
        return RouteCache([], packing[:], [], [], [], [], {}, 0.0, 0.0, 0.0, 0.0)

    distance_matrix = _distance_matrix(inst)
    speed_max = float(inst.max_speed)
    speed_min = float(inst.min_speed)
    capacity = float(inst.capacity)
    rent_per_time = float(inst.rent_per_time)

    speed_coef = (speed_max - speed_min) / capacity if capacity > 0 else 0.0

    weight_by_position = [0.0] * n_positions
    time_by_position = [0.0] * n_positions
    accumulated_weight = [0.0] * n_positions
    accumulated_time = [0.0] * n_positions

    pos_in_tour = {city: index for index, city in enumerate(tour)}

    # Fase 1: calcula peso recogido en cada ciudad y peso acumulado por posición.
    carried_weight = 0.0
    for index in range(n_positions):
        city_id = tour[index]
        local_weight = city_items_taken_weight(inst, packing, city_id)

        weight_by_position[index] = local_weight
        carried_weight += local_weight
        accumulated_weight[index] = carried_weight

    # Fase 2: calcula tiempos por arco usando el peso al salir de cada ciudad.
    total_time = 0.0

    for index in range(n_positions - 1):
        curr_weight = accumulated_weight[index]

        speed_curr = speed_max - (curr_weight * speed_coef)
        if speed_curr < speed_min:
            speed_curr = speed_min

        curr_city = tour[index]
        next_city = tour[index + 1]
        travel_time = float(distance_matrix[curr_city, next_city]) / speed_curr

        time_by_position[index] = travel_time
        total_time += travel_time
        accumulated_time[index] = total_time

    # Arco de cierre: última ciudad -> primera ciudad.
    curr_weight = accumulated_weight[n_positions - 1]
    speed_curr = speed_max - (curr_weight * speed_coef)
    if speed_curr < speed_min:
        speed_curr = speed_min

    curr_city = tour[n_positions - 1]
    next_city = tour[0]
    travel_time = float(distance_matrix[curr_city, next_city]) / speed_curr

    time_by_position[n_positions - 1] = travel_time
    total_time += travel_time
    accumulated_time[n_positions - 1] = total_time

    # Fase 3: métricas finales de la solución.
    items = inst.items
    total_profit = float(
        sum(items[item_id].profit for item_id, selected in enumerate(packing) if selected)
    )

    final_weight = accumulated_weight[-1]
    final_time = accumulated_time[-1]
    objective = total_profit - rent_per_time * final_time

    return RouteCache(
        tour[:],
        packing[:],
        accumulated_time,
        accumulated_weight,
        time_by_position,
        weight_by_position,
        pos_in_tour,
        total_profit,
        final_weight,
        final_time,
        objective,
    )


def precompute_remaining_distances(inst: Any, tour: list[int]) -> list[float]:
    """Calcula la distancia restante desde cada posición del tour.

    Para cada posición ``i``, retorna la distancia acumulada desde ``tour[i]``
    hasta completar el ciclo y volver a la ciudad inicial. La distancia incluye
    el arco de cierre ``tour[-1] -> tour[0]``.

    Args:
        inst: Instancia TTP.
        tour: Secuencia de ciudades del tour.

    Returns:
        Lista donde ``out[i]`` contiene la distancia restante desde ``tour[i]``
        hasta cerrar el ciclo.
    """
    n_positions = len(tour)
    if n_positions == 0:
        return []

    remaining_distances = [0.0] * n_positions
    accumulated_distance = 0.0
    distance_matrix = _distance_matrix(inst)

    # Primero se considera el arco de cierre: última ciudad -> primera ciudad.
    last_city = tour[n_positions - 1]
    first_city = tour[0]
    accumulated_distance += float(distance_matrix[last_city, first_city])
    remaining_distances[n_positions - 1] = accumulated_distance

    # Luego se avanza hacia atrás sumando cada arco tour[i] -> tour[i + 1].
    for index in range(n_positions - 2, -1, -1):
        city_a = tour[index]
        city_b = tour[index + 1]

        accumulated_distance += float(distance_matrix[city_a, city_b])
        remaining_distances[index] = accumulated_distance

    return remaining_distances


# =============================================================================
# Movimientos y actualizaciones rápidas
# =============================================================================


def apply_2opt_inplace(cache: RouteCache, i: int, j: int) -> None:
    """Aplica un movimiento 2-opt directamente sobre el tour cacheado.

    Invierte el segmento ``tour[i + 1:j + 1]`` y actualiza ``pos_in_tour``.
    Esta función no recalcula tiempos ni pesos; si el movimiento se acepta,
    la caché física debe reconstruirse después.

    Args:
        cache: Caché de ruta a modificar.
        i: Índice del primer corte.
        j: Índice del segundo corte.
    """
    tour = cache.tour

    # Invierte el tramo interno del 2-opt; no toca métricas físicas.
    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])

    # Actualiza el mapa inverso para el tramo afectado.
    for index in range(i, j + 1):
        cache.pos_in_tour[tour[index]] = index


def eval_time_after_2opt(
    inst: Any,
    cache: RouteCache,
    i: int,
    j: int,
) -> float:
    """Evalúa el tiempo total después de un movimiento 2-opt hipotético.

    El movimiento invierte el segmento ``tour[i + 1:j + 1]``. La función no
    modifica la caché; combina:
        1. tiempo del prefijo no afectado;
        2. tiempo del bloque invertido;
        3. tiempo del sufijo no afectado.

    Args:
        inst: Instancia TTP.
        cache: Estado actual de la ruta.
        i: Índice inicial del corte 2-opt.
        j: Índice final del corte 2-opt.

    Returns:
        Tiempo total estimado después del movimiento.
    """
    tour = cache.tour
    n_positions = len(tour)

    if n_positions == 0:
        return 0.0

    distance_matrix = _distance_matrix(inst)
    speed_max = float(inst.max_speed)
    speed_min = float(inst.min_speed)
    capacity = float(inst.capacity)
    speed_coef = (speed_max - speed_min) / capacity if capacity > 0 else 0.0

    # Prefijo y sufijo permanecen iguales; solo se recalcula el bloque invertido.
    prefix_time = cache.t_acc[i - 1] if i > 0 else 0.0
    suffix_time = cache.t_acc[-1] - cache.t_acc[j]

    carried_weight = cache.w_acc[i]
    block_time = 0.0

    city_a = tour[i]
    city_b = tour[j]

    speed_curr = speed_max - carried_weight * speed_coef
    if speed_curr < speed_min:
        speed_curr = speed_min

    # Nuevo arco de entrada al segmento invertido.
    block_time += float(distance_matrix[city_a, city_b]) / speed_curr

    weight_by_position = cache.w_reg

    # Recorre el segmento invertido desde j hacia i + 1.
    for index in range(j, i + 1, -1):
        carried_weight += weight_by_position[index]

        target_index = index - 1
        node_index = tour[index]
        node_target = tour[target_index]

        speed_curr = speed_max - carried_weight * speed_coef
        if speed_curr < speed_min:
            speed_curr = speed_min

        block_time += float(distance_matrix[node_index, node_target]) / speed_curr

    index = i + 1
    carried_weight += weight_by_position[index]

    next_index = j + 1
    if next_index >= n_positions:
        next_index = 0

    node_index = tour[index]
    node_target = tour[next_index]

    speed_curr = speed_max - carried_weight * speed_coef
    if speed_curr < speed_min:
        speed_curr = speed_min

    # Nuevo arco de salida del segmento invertido.
    block_time += float(distance_matrix[node_index, node_target]) / speed_curr

    return prefix_time + block_time + suffix_time


def incremental_bitflip(
    inst: Any,
    cache: RouteCache,
    item_id: int,
    new_value: int,
) -> None:
    """Actualiza la caché al cambiar la selección de un ítem.

    Recalcula únicamente los pesos y tiempos afectados desde la ciudad donde se
    encuentra el ítem modificado hasta el cierre del tour. Esto evita reconstruir
    toda la caché para cada bit-flip probado por KRP.

    Args:
        inst: Instancia TTP.
        cache: Caché actual de la ruta.
        item_id: ID del ítem modificado.
        new_value: Nuevo valor binario del ítem, ``1`` para tomar y ``0`` para
            dejar.
    """
    old_value = cache.packing[item_id]
    if old_value == new_value:
        return

    item = inst.items[item_id]
    item_weight = float(item.weight)
    item_profit = float(item.profit)

    distance_matrix = _distance_matrix(inst)
    speed_max = float(inst.max_speed)
    speed_min = float(inst.min_speed)
    capacity = float(inst.capacity)
    speed_coef = (speed_max - speed_min) / capacity if capacity > 0 else 0.0

    # El cambio de peso afecta desde la ciudad del ítem hasta cerrar el ciclo.
    item_position = cache.pos_in_tour[item.city_id]

    cache.packing[item_id] = new_value

    delta_weight = item_weight if new_value == 1 else -item_weight
    delta_profit = item_profit if new_value == 1 else -item_profit

    cache.g_profit += delta_profit
    cache.g_weight += delta_weight
    cache.w_reg[item_position] += delta_weight

    tour = cache.tour
    accumulated_weight = cache.w_acc
    time_by_position = cache.t_reg
    accumulated_time = cache.t_acc

    n_positions = len(tour)
    current_accumulated_time = accumulated_time[item_position - 1] if item_position > 0 else 0.0
    total_time_diff = 0.0

    last_position = n_positions - 1
    start_index = item_position

    # Recalcula arcos desde la posición afectada hasta la penúltima posición.
    if start_index < last_position:
        for index in range(start_index, last_position):
            accumulated_weight[index] += delta_weight
            curr_weight = accumulated_weight[index]

            speed_curr = speed_max - curr_weight * speed_coef
            if speed_curr < speed_min:
                speed_curr = speed_min

            city_a = tour[index]
            city_b = tour[index + 1]
            new_travel_time = float(distance_matrix[city_a, city_b]) / speed_curr

            total_time_diff += new_travel_time - time_by_position[index]
            time_by_position[index] = new_travel_time

            current_accumulated_time += new_travel_time
            accumulated_time[index] = current_accumulated_time

        start_index = last_position

    # Recalcula el arco de cierre: última ciudad -> primera ciudad.
    if start_index == n_positions - 1:
        index = n_positions - 1
        accumulated_weight[index] += delta_weight
        curr_weight = accumulated_weight[index]

        speed_curr = speed_max - curr_weight * speed_coef
        if speed_curr < speed_min:
            speed_curr = speed_min

        city_a = tour[index]
        city_b = tour[0]
        new_travel_time = float(distance_matrix[city_a, city_b]) / speed_curr

        total_time_diff += new_travel_time - time_by_position[index]
        time_by_position[index] = new_travel_time

        current_accumulated_time += new_travel_time
        accumulated_time[index] = current_accumulated_time

    cache.f_time += total_time_diff
    cache.G_gain = cache.g_profit - float(inst.rent_per_time) * cache.f_time