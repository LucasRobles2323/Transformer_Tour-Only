#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/evaluation/baselines.py

"""Baselines clásicos para evaluación de soluciones TTP.

Este módulo construye y evalúa tours base, principalmente un baseline TSP puro
que ignora los ítems y luego optimiza el packing sobre el tour fijo.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.ttp_packages.domain.tour_ops import normalize_tour

from .config import DEFAULT_N_RESTARTS, DEFAULT_START_CITY, DEFAULT_TIME_BUDGET_S
from .fixed_tour import evaluate_fixed_tour_ttp


def build_tsp_baseline_tour(
    instance: Any,
    start_city: int = DEFAULT_START_CITY,
) -> list[int]:
    """Construye un tour baseline resolviendo el TSP sobre las ciudades.

    El tour ignora los ítems y sirve como referencia de ruta corta. Luego puede
    evaluarse bajo el objetivo TTP optimizando packing sobre ese tour fijo.

    Args:
        instance: Instancia TTP a evaluar.
        start_city: Ciudad inicial usada para normalizar el tour.

    Returns:
        Tour TSP normalizado.
    """
    # Import local: OR-Tools solo se necesita al construir este baseline.
    from src.ttp_packages.optimization.classical.tsp.tsp_api import (
        solve_tsp_with_ortools,
    )

    tour = solve_tsp_with_ortools(instance)
    return normalize_tour(
        tour,
        n_cities=int(instance.n_cities),
        start_city=start_city,
    )


def evaluate_tsp_baseline(
    instance: Any,
    *,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    n_restarts: int = DEFAULT_N_RESTARTS,
    seed: Optional[int] = None,
    start_city: int = DEFAULT_START_CITY,
) -> Dict[str, Any]:
    """Evalúa el baseline TSP bajo el objetivo TTP.

    Primero construye un tour TSP puro y luego optimiza el packing sobre ese
    tour fijo para obtener una solución TTP comparable.

    Args:
        instance: Instancia TTP a evaluar.
        time_budget_s: Presupuesto de tiempo para optimizar packing.
        n_restarts: Cantidad de reinicios del solver de packing.
        seed: Semilla opcional para el solver.
        start_city: Ciudad inicial usada para normalizar el tour.

    Returns:
        Diccionario con el tour, la solución y métricas principales.
    """
    tour = build_tsp_baseline_tour(instance, start_city=start_city)
    sol = evaluate_fixed_tour_ttp(
        instance,
        tour,
        start_city=start_city,
        time_budget_s=time_budget_s,
        n_restarts=n_restarts,
        seed=seed,
    )

    return {
        "tour": tour,
        "solution": sol,
        "objective": float(sol.objective),
        "profit": float(sol.profit),
        "time": float(sol.time),
    }