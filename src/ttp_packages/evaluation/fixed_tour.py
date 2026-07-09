#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/evaluation/fixed_tour.py

"""Evaluación TTP de tours fijos.

Este módulo valida un tour dado y delega al solver de packing la construcción
de la mejor selección de ítems para ese orden de ciudades.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from src.ttp_packages.domain.solution import TTPSolution
from src.ttp_packages.domain.tour_ops import validate_tour
from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.optimization.classical.packing.api import solve_pack_for_fixed_tour

from .config import (
    DEFAULT_JITTER,
    DEFAULT_N_RESTARTS,
    DEFAULT_START_CITY,
    DEFAULT_TIME_BUDGET_S,
)


logger = setup_logger(__name__)


def evaluate_fixed_tour_ttp(
    inst: Any,
    tour: List[int],
    *,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    n_restarts: int = DEFAULT_N_RESTARTS,
    seed: Optional[int] = None,
    jitter: float = DEFAULT_JITTER,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
    start_city: int = DEFAULT_START_CITY,
) -> TTPSolution:
    """Evalúa un tour fijo bajo el objetivo TTP.

    Valida estructuralmente el tour, lo normaliza y luego resuelve el packing
    asociado mediante el solver clásico configurado.

    Args:
        inst: Instancia TTP.
        tour: Tour a evaluar.
        time_budget_s: Presupuesto de tiempo para el solver de packing.
        n_restarts: Cantidad de reinicios del solver.
        seed: Semilla opcional.
        jitter: Perturbación opcional usada por el solver.
        verbose: Si es True, habilita salida detallada del solver.
        log_fn: Función opcional de logging.
        start_city: Ciudad inicial esperada para normalizar el tour.

    Returns:
        Mejor solución TTP encontrada para el tour fijo.

    Raises:
        ValueError: Si el tour no es válido.
    """
    if log_fn is None:
        log_fn = logger.info

    check = validate_tour(inst, tour, start_city=start_city)
    if not check["is_valid"]:
        logger.error("Tour inválido detectado: %s", check["issues"])
        raise ValueError(f"Tour inválido: {check['issues']}")

    # El solver recibe siempre el tour normalizado para evitar inconsistencias.
    normalized_tour = check["normalized_tour"]

    return solve_pack_for_fixed_tour(
        inst=inst,
        tour=normalized_tour,
        time_budget_s=time_budget_s,
        n_restarts=n_restarts,
        seed=seed,
        jitter=jitter,
        verbose=verbose,
        log_fn=log_fn,
    )