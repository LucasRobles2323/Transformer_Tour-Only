#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/packing/api.py

"""API pública para optimizar packing sobre un tour fijo.

Este módulo combina una construcción inicial heurística con búsqueda local para
obtener una solución TTP completa cuando el orden de visita de ciudades ya está
definido.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, Optional

from src.ttp_packages.domain.solution import TTPSolution

from .config import (
    FIXED_TOUR_PACKING_JITTER,
    FIXED_TOUR_PACKING_RESTARTS,
    FIXED_TOUR_PACKING_TIME_S,
)
from .initial_packing import build_initial_packing_for_fixed_tour
from .local_search import improve_packing_for_fixed_tour


def solve_pack_for_fixed_tour(
    inst: Any,
    tour: list[int],
    *,
    time_budget_s: float = FIXED_TOUR_PACKING_TIME_S,
    n_restarts: int = FIXED_TOUR_PACKING_RESTARTS,
    seed: Optional[int] = None,
    jitter: float = FIXED_TOUR_PACKING_JITTER,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> TTPSolution:
    """Optimiza el packing asumiendo un tour fijo.

    Ejecuta varios reinicios independientes. En cada reinicio construye un
    packing inicial heurístico y luego lo mejora mediante búsqueda local.

    Args:
        inst: Instancia TTP.
        tour: Tour fijo sobre el cual optimizar el packing.
        time_budget_s: Presupuesto total de tiempo para todos los reinicios.
        n_restarts: Cantidad de reinicios independientes.
        seed: Semilla base opcional.
        jitter: Ruido usado para desempatar candidatos en la construcción inicial.
        verbose: Si es True, muestra logs detallados durante la búsqueda local.
        log_fn: Función opcional de logging.

    Returns:
        Mejor solución TTP encontrada para el tour fijo.
    """
    n_items = int(inst.m_items)
    if n_items <= 0:
        return TTPSolution(inst, tour[:], [0] * n_items)

    base_seed = 0 if seed is None else int(seed)
    best_solution: Optional[TTPSolution] = None
    best_objective = float("-inf")

    total_restarts = max(1, int(n_restarts))
    time_per_restart = max(float(time_budget_s), 1e-6) / total_restarts

    for restart_index in range(total_restarts):
        # Cada reinicio usa semillas separadas para construcción inicial y mejora.
        rng = random.Random(base_seed + 10007 * restart_index + 17)

        initial_packing = build_initial_packing_for_fixed_tour(
            inst,
            tour,
            rng=rng,
            jitter=float(jitter),
        )

        improved_packing = improve_packing_for_fixed_tour(
            inst,
            tour,
            initial_packing,
            time_budget_s=time_per_restart,
            seed=base_seed + 7919 * restart_index + 31,
            verbose=verbose,
            log_fn=log_fn,
        )

        candidate_solution = TTPSolution(inst, tour[:], improved_packing[:])

        if candidate_solution.objective > best_objective:
            best_objective = float(candidate_solution.objective)
            best_solution = candidate_solution

    # Fallback defensivo: no debería ocurrir porque total_restarts >= 1.
    if best_solution is None:
        best_solution = TTPSolution(inst, tour[:], [0] * n_items)

    return best_solution