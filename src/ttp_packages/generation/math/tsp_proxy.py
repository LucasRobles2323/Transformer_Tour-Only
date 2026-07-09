#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/generation/math/tsp_proxy.py

from __future__ import annotations

from typing import Callable, List, Optional

from src.ttp_packages.domain.instance import TTPInstance
from src.ttp_packages.infrastructure.logging import setup_logger

from .config import DEFAULT_2OPT_ITERS, DEFAULT_START_NODE

# Inicialización del logger para este módulo específico
logger = setup_logger(__name__)

def build_tour_nn_2opt(
    inst: TTPInstance, 
    start_node: int = DEFAULT_START_NODE, 
    two_opt_iters: int = DEFAULT_2OPT_ITERS,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[int]:
    """Genera un tour geométricamente optimizado para el TSP.

    Combina la heurística de 'Vecino más Cercano' (Nearest Neighbor) para una 
    construcción rápida y el algoritmo '2-Opt' para eliminar cruces en el tour.

    Args:
        inst: Instancia del problema TTP que contiene las ciudades y distancias.
        start_node: Índice de la ciudad donde comienza el recorrido.
        two_opt_iters: Límite máximo de iteraciones para la mejora 2-Opt.

    Returns:
        List[int]: Una lista ordenada de índices de ciudades que representan el tour.
    """
    if log_fn is None:
        log_fn = logger.info

    # Aseguramos que exista la matriz de distancias antes de usarla
    if inst.distance_matrix is None:
        if verbose:
            log_fn("Matriz de distancias no encontrada. Generándola...")
        inst.create_distance_matrix()

    dm = inst.distance_matrix
    if dm is None:
        raise RuntimeError("No se pudo crear la matriz de distancias.")

    n = int(inst.n_cities)

    if n < 2:
        logger.warning("Instancia con solo %s ciudades. Retornando tour trivial.", n)
        return list(range(n))

    if not 0 <= start_node < n:
        raise ValueError(f"start_node={start_node} fuera de rango para n_cities={n}.")

    # --- Fase 1: Construcción con Nearest Neighbor ---
    # Mantiene un conjunto de nodos no visitados y elige el más cercano al actual
    unvisited = set(range(n))
    unvisited.remove(start_node)

    tour = [start_node]
    curr = start_node

    while unvisited:
        # Elegimos el nodo j con menor distancia desde curr
        nxt = min(unvisited, key=lambda j: float(dm[curr, j]))
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt

    if verbose: 
        log_fn(f"Tour inicial NN construido para {n} ciudades.")

    # --- Fase 2: Mejora con 2-Opt ---
    # Optimización: calculamos el delta de distancia en O(1)
    def calculate_delta_dist(i, k, t):
        """Calcula el cambio en la distancia si se intercambian dos aristas."""
        a, b = t[i], t[(i + 1) % n]
        c, d = t[k], t[(k + 1) % n]

        current_dist = float(dm[a, b] + dm[c, d])
        new_dist = float(dm[a, c] + dm[b, d])

        return new_dist - current_dist

    improved = True
    iterations = 0

    two_opt_iters = max(0, int(two_opt_iters))
    
    # Repetimos mientras haya mejoras y no superemos el límite de iteraciones
    while improved and iterations < two_opt_iters:
        improved = False
        iterations += 1

        for i in range(n - 1):
            # Evitamos aristas adyacentes y rompimiento del cierre del tour
            for k in range(i + 2, n - (0 if i > 0 else 1)):
                if calculate_delta_dist(i, k, tour) < 0:
                    # Inversión del segmento intermedio (i+1..k)
                    tour[i + 1:k + 1] = reversed(tour[i + 1:k + 1])
                    improved = True

    if verbose: 
        log_fn(f"Tour finalizado tras {iterations} iteraciones de 2-Opt.")
    return tour