#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/evaluation/compare.py

"""Comparación de tours predichos contra baselines y soluciones heurísticas.

Este módulo evalúa un tour producido por un modelo frente a un baseline TSP y
una solución CS2SAR ya calculada, usando métricas de objetivo, gap, distancia y
solapamiento de aristas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.ttp_packages.domain.solution import TTPSolution
from src.ttp_packages.domain.tour_ops import validate_tour

from .config import DEFAULT_N_RESTARTS, DEFAULT_START_CITY, DEFAULT_TIME_BUDGET_S
from .tour_metrics import compute_relative_gap, tour_distance, tour_edge_set


def evaluate_model_tour(
    instance: Any,
    pred_tour: List[int],
    cs2sar_solution: TTPSolution,
    *,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    n_restarts: int = DEFAULT_N_RESTARTS,
    start_city: int = DEFAULT_START_CITY,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compara un tour predicho contra TSP y CS2SAR.

    Valida los tours, optimiza packing para el tour predicho y para el baseline
    TSP, y calcula métricas comparativas contra TSP y CS2SAR.

    Args:
        instance: Instancia TTP evaluada.
        pred_tour: Tour predicho por el modelo.
        cs2sar_solution: Solución CS2SAR ya evaluada.
        time_budget_s: Presupuesto de tiempo para TSP y packing.
        n_restarts: Cantidad de reinicios del solver de packing.
        start_city: Ciudad inicial esperada en los tours.
        seed: Semilla opcional para el solver de packing.

    Returns:
        Diccionario con validez de tours y métricas comparativas. Si algún tour
        no es válido, retorna solo la información de validez.
    """
    # Import local: los solvers clásicos pueden cargar dependencias pesadas.
    from src.ttp_packages.optimization.classical.packing.api import (
        solve_pack_for_fixed_tour,
    )
    from src.ttp_packages.optimization.classical.tsp.tsp_api import (
        solve_tsp_with_ortools,
    )

    raw_tsp_tour = solve_tsp_with_ortools(
        instance,
        time_limit=time_budget_s,
        verbose=False,
    )

    val_pred = validate_tour(instance, pred_tour, start_city=start_city)
    val_tsp = validate_tour(instance, raw_tsp_tour, start_city=start_city)
    val_cs2sar = validate_tour(instance, cs2sar_solution.tour, start_city=start_city)

    all_valid = (
        val_pred["is_valid"]
        and val_tsp["is_valid"]
        and val_cs2sar["is_valid"]
    )

    out: Dict[str, Any] = {
        "all_tours_valid": all_valid,
        "pred_tour_issues": val_pred["issues"],
    }

    # Si algún tour es inválido, evita calcular métricas sobre soluciones inconsistentes.
    if not all_valid:
        return out

    t_pred = val_pred["normalized_tour"]
    t_tsp = val_tsp["normalized_tour"]
    t_cs2sar = val_cs2sar["normalized_tour"]

    # CS2SAR ya viene con packing y objetivo evaluados; solo se resuelve predicción y TSP.
    sol_pred = solve_pack_for_fixed_tour(
        instance,
        t_pred,
        time_budget_s=time_budget_s,
        n_restarts=n_restarts,
        seed=seed,
    )
    sol_tsp = solve_pack_for_fixed_tour(
        instance,
        t_tsp,
        time_budget_s=time_budget_s,
        n_restarts=n_restarts,
        seed=seed,
    )

    model_obj = float(sol_pred.objective)
    tsp_obj = float(sol_tsp.objective)
    cs2sar_obj = float(cs2sar_solution.objective)

    pred_edges = tour_edge_set(t_pred)
    cs2sar_edges = tour_edge_set(t_cs2sar)

    # Evita división por cero en instancias degeneradas o tours vacíos.
    edge_overlap = len(pred_edges & cs2sar_edges) / max(len(cs2sar_edges), 1)

    dist_pred = tour_distance(instance, t_pred)
    dist_tsp = tour_distance(instance, t_tsp)
    dist_penalty = (dist_pred / dist_tsp) - 1.0 if dist_tsp > 0 else 0.0

    out.update(
        {
            # Objetivos Absolutos
            "model_obj": model_obj,
            "tsp_obj": tsp_obj,
            "cs2sar_obj": cs2sar_obj,
            # Comparación vs CS2SAR (Oracle)
            "rel_gap_cs2sar": compute_relative_gap(model_obj, cs2sar_obj),
            "edge_overlap_cs2sar": float(edge_overlap),
            # Comparación vs TSP (Baseline)
            "gain_vs_tsp": model_obj - tsp_obj,
            "rel_gap_tsp": compute_relative_gap(model_obj, tsp_obj),
            "tsp_distance_penalty": float(dist_penalty),
            "win_vs_tsp": bool(model_obj > tsp_obj),
        }
    )

    return out