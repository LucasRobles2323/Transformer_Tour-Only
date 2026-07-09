#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/tsp/tsp_api.py

"""API OR-Tools para resolver TSP puro sobre instancias TTP.

Este módulo usa el solver de ruteo de Google OR-Tools para construir un tour
sobre las ciudades, ignorando los ítems del problema TTP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.ttp_packages.infrastructure.logging import setup_logger

from .config import (
    TSP_DEFAULT_TIME_LIMIT,
    TSP_DEFAULT_VEHICLES,
    TSP_DEPOT_INDEX,
    TSP_DISTANCE_SCALING_FACTOR,
)

if TYPE_CHECKING:
    from src.ttp_packages.domain.instance import TTPInstance


logger = setup_logger(__name__)


def _build_scaled_distance_matrix(inst: "TTPInstance") -> np.ndarray:
    """Construye matriz de distancias enteras para OR-Tools.

    OR-Tools espera costos enteros. Por eso se calcula distancia euclidiana y se
    multiplica por ``TSP_DISTANCE_SCALING_FACTOR``.

    Args:
        inst: Instancia TTP con ciudades y coordenadas.

    Returns:
        Matriz ``(N, N)`` de costos enteros escalados.
    """
    coords = np.asarray(
        [(city.x, city.y) for city in inst.cities],
        dtype=np.float64,
    )

    # Broadcasting para calcular todas las distancias euclidianas par-a-par.
    diffs = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))

    return (distances * TSP_DISTANCE_SCALING_FACTOR).astype(np.int64)


def solve_tsp_with_ortools(
    inst: "TTPInstance",
    time_limit: float = TSP_DEFAULT_TIME_LIMIT,
    verbose: bool = False,
) -> list[int]:
    """Resuelve el TSP puro usando Google OR-Tools.

    El solver ignora ítems y optimiza solo la distancia entre ciudades. La salida
    es un tour abierto en forma de permutación; no repite el depósito al final.

    Args:
        inst: Instancia TTP con ciudades y coordenadas.
        time_limit: Tiempo máximo de optimización en segundos.
        verbose: Si es True, registra información de la corrida.

    Returns:
        Tour TSP como lista de identificadores de ciudades.

    Raises:
        ImportError: Si OR-Tools no está instalado.
    """
    n_cities = int(inst.n_cities)

    if n_cities < 2:
        return [0] if n_cities == 1 else []

    # Import local: OR-Tools solo se necesita cuando se ejecuta este solver.
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    scaled_distance_matrix = _build_scaled_distance_matrix(inst)

    # RoutingIndexManager traduce entre índices internos de OR-Tools y nodos reales.
    manager = pywrapcp.RoutingIndexManager(
        n_cities,
        TSP_DEFAULT_VEHICLES,
        TSP_DEPOT_INDEX,
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        """Devuelve el costo entero escalado entre dos índices OR-Tools."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return int(scaled_distance_matrix[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    # Mantiene comportamiento original: OR-Tools recibe segundos enteros.
    search_parameters.time_limit.seconds = int(time_limit)

    if verbose:
        logger.info("Iniciando OR-Tools para %s ciudades...", n_cities)

    assignment = routing.SolveWithParameters(search_parameters)

    if not assignment:
        logger.warning("No se encontró solución. Retornando orden secuencial.")
        return list(range(n_cities))

    tour: list[int] = []
    index = routing.Start(0)

    # Se recorre la solución siguiendo NextVar hasta llegar al nodo final.
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = assignment.Value(routing.NextVar(index))

    if verbose:
        actual_cost = assignment.ObjectiveValue() / TSP_DISTANCE_SCALING_FACTOR
        logger.debug(
            "[OR-Tools] Búsqueda finalizada. Costo total estimado: %.2f",
            actual_cost,
        )

    return tour