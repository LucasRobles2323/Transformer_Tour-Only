#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/solve_instance.py

"""Resolución de instancias TTP desde la capa de aplicación.

Este módulo encapsula la creación y ejecución del solver CS2SA-R para que los
scripts no dependan directamente del paquete ``optimization``.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from . import config as cfg

from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.optimization.classical.ttp.cs2sa_r.api import CS2SARSolver


logger = setup_logger(__name__)


def solve_with_cs2sar(
    instance,
    time_to_solve: float = 60.0,
    *,
    restart_mode: str = cfg.DEFAULT_MODE_RESTART,
    no_improve_patience: int = cfg.DEFAULT_NO_IMPROVE_PATIENCE,
    seed: Optional[int] = None,
    verbose_sections: Optional[Iterable[str] | str] = cfg.DEFAULT_SOLVER_VERBOSE_SECTIONS,
    verify_integrity: bool = cfg.DEFAULT_SOLVER_VERIFY_INTEGRITY,
    log_fn: Optional[Callable[[str], None]] = None,
):
    """Resuelve una instancia TTP con CS2SA-R.

    Args:
        instance: Instancia TTP a resolver.
        time_to_solve: Presupuesto de tiempo en segundos.
        restart_mode: Modo de reinicio del solver.
        no_improve_patience: Ciclos sin mejora antes de reiniciar.
        seed: Semilla opcional para reproducibilidad.
        verbose_sections: Secciones de logging del solver.
        verify_integrity: Si es ``True``, ejecuta validaciones internas del
            solver al finalizar.
        log_fn: Función opcional de logging.

    Returns:
        Solución TTP retornada por CS2SA-R.

    Raises:
        ValueError: Si ``time_to_solve`` no es positivo.
    """
    if time_to_solve <= 0:
        raise ValueError("time_to_solve debe ser > 0")

    run_seed = cfg.resolve_seed(seed)

    solver = CS2SARSolver(
        instance,
        restart_mode=restart_mode,
        no_improve_patience=no_improve_patience,
        seed=run_seed,
    )

    return solver.solve(
        time_budget_s=float(time_to_solve),
        verbose_sections=verbose_sections,
        verify_integrity=bool(verify_integrity),
        log_fn=log_fn,
    )