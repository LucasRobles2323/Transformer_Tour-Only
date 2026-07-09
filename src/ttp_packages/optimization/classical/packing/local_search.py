#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/packing/local_search.py

"""Búsqueda local para mejorar packing sobre un tour fijo.

Este módulo delega la mejora a ``KRPOptimizer`` manteniendo fijo el tour y
optimizando únicamente la selección de ítems.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Optional

from src.ttp_packages.domain.tour_ops import canonicalize_tour
from src.ttp_packages.infrastructure.logging import setup_logger

from .config import FIXED_TOUR_PACKING_TIME_S


logger = setup_logger(__name__)


def improve_packing_for_fixed_tour(
    inst: Any,
    tour: list[int],
    packing: list[int],
    *,
    time_budget_s: float = FIXED_TOUR_PACKING_TIME_S,
    seed: Optional[int] = None,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[int]:
    """Mejora un packing sobre un tour fijo usando ``KRPOptimizer``.

    Args:
        inst: Instancia TTP.
        tour: Tour base.
        packing: Packing inicial a mejorar.
        time_budget_s: Presupuesto máximo de tiempo para la mejora.
        seed: Semilla opcional del optimizador.
        verbose: Si es True, muestra detalles de optimización.
        log_fn: Función opcional de logging.

    Returns:
        Packing optimizado como lista binaria.
    """
    if log_fn is None:
        log_fn = logger.info

    n_cities = int(inst.n_cities)
    canonical_tour = canonicalize_tour(tour, n_cities=n_cities, start_city=0)

    # Import local: KRPOptimizer solo se necesita al ejecutar la búsqueda local.
    from src.ttp_packages.optimization.classical.ttp.cs2sa_r.krp_optimizer import (
        KRPOptimizer,
    )

    optimizer = KRPOptimizer(rnd_seed=(1 if seed is None else int(seed)))
    deadline = time.time() + float(max(time_budget_s, 1e-6))

    _tour_out, improved_packing = optimizer.optimize(
        inst,
        canonical_tour,
        list(map(int, packing)),
        verbose=verbose,
        log_fn=log_fn,
        deadline=deadline,
    )

    return list(map(int, improved_packing))