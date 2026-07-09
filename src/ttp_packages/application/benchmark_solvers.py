#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/benchmark_solvers.py

"""Benchmarks de solvers clásicos desde la capa de aplicación.

Este módulo evalúa CS2SA-R con múltiples presupuestos de tiempo y compara
CS2SA-R contra el baseline TSP+KRP. Los workflows retornan resultados listos
para ser consumidos por scripts CLI.
"""

from __future__ import annotations

import random
from string import ascii_uppercase
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.ttp_packages.evaluation.benchmarks import (
    EVAL_BIG_SEPARATED,
    EVAL_CS2SAR_ITERATIONS,
    EVAL_INST_FNAMES,
    EVAL_LIST_TIME_TO_SOL,
    EVAL_SEPARATED,
)
from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.infrastructure.storage.instance_io import load_instance
from src.ttp_packages.optimization.classical.ttp.cs2sa_r.api import CS2SARSolver


logger = setup_logger(__name__)


def _mean_or_none(values: List[float]) -> Optional[float]:
    """Calcula el promedio de una lista.

    Args:
        values: Valores numéricos.

    Returns:
        Promedio como ``float`` o ``None`` si la lista está vacía.
    """
    if not values:
        return None

    return float(sum(values) / len(values))


def _fmt_optional_float(value: Optional[float], precision: int = 4) -> str:
    """Formatea un ``float`` opcional para logging.

    Args:
        value: Valor numérico opcional.
        precision: Cantidad de decimales.

    Returns:
        Texto formateado o ``"NA"`` si ``value`` es ``None``.
    """
    if value is None:
        return "NA"

    return f"{value:.{precision}f}"


def evaluate_solver_cs2sar(
    iterations: int = EVAL_CS2SAR_ITERATIONS,
    instances_fnames_to_evaluate: List[str] = EVAL_INST_FNAMES,
    time_solutions: List[int] = EVAL_LIST_TIME_TO_SOL,
    shuffle: bool = True,
    verbose_sections: Optional[Iterable[str] | str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evalúa CS2SA-R en múltiples instancias y presupuestos de tiempo.

    Args:
        iterations: Máximo de instancias a evaluar.
        instances_fnames_to_evaluate: Nombres de instancias a evaluar.
        time_solutions: Presupuestos de tiempo por corrida.
        shuffle: Si es ``True``, baraja el orden de instancias.
        verbose_sections: Secciones de logging reenviadas a
            ``CS2SARSolver.solve()``.
        log_fn: Función de logging. Si es ``None``, usa ``logger.info``.

    Returns:
        Diccionario con resultados por instancia y resumen por presupuesto.
    """
    if log_fn is None:
        log_fn = logger.info

    labels = ascii_uppercase[: len(time_solutions)]

    instances_fnames = list(instances_fnames_to_evaluate)
    if shuffle:
        random.shuffle(instances_fnames)

    results: List[Dict[str, Any]] = []
    by_budget: Dict[int, List[float]] = {
        time_budget: [] for time_budget in time_solutions
    }
    by_budget_time: Dict[int, List[float]] = {
        time_budget: [] for time_budget in time_solutions
    }

    for instance_index, file_name in enumerate(instances_fnames):
        if instance_index >= iterations:
            break

        log_fn(EVAL_BIG_SEPARATED)
        log_fn(f"\nInstancia: {file_name}")

        instance = load_instance(fname=file_name)

        log_fn(EVAL_SEPARATED)
        log_fn("Soluciones:")

        solutions: Dict[int, Any] = {}
        elapsed_times: Dict[int, float] = {}

        for label, time_budget in zip(labels, time_solutions):
            solver = CS2SARSolver(instance, seed=12345)

            start_time = perf_counter()
            solution = solver.solve(
                time_budget_s=time_budget,
                verbose_sections=verbose_sections,
                log_fn=log_fn,
            )
            elapsed = perf_counter() - start_time

            solutions[time_budget] = solution
            elapsed_times[time_budget] = elapsed
            by_budget[time_budget].append(float(solution.objective))
            by_budget_time[time_budget].append(float(elapsed))

            log_fn(
                f"  {label}.  sol{label} "
                f"(budget={time_budget:.2f}s, time={elapsed:.2f}s) -> "
                f"obj={solution.objective}"
            )

        results.append(
            {
                "instance": file_name,
                "solutions": {
                    time_budget: {
                        "label": label,
                        "objective": float(solutions[time_budget].objective),
                        "elapsed_s": float(elapsed_times[time_budget]),
                    }
                    for label, time_budget in zip(labels, time_solutions)
                },
            }
        )

    global_summary: Dict[str, Any] = {
        "n_instances": len(results),
        "by_budget": {},
    }

    log_fn(EVAL_BIG_SEPARATED)
    log_fn("\nRESUMEN GLOBAL")
    log_fn(EVAL_SEPARATED)

    for label, time_budget in zip(labels, time_solutions):
        objectives = by_budget[time_budget]
        elapsed_values = by_budget_time[time_budget]

        mean_objective = _mean_or_none(objectives)
        best_objective = float(max(objectives)) if objectives else None
        worst_objective = float(min(objectives)) if objectives else None
        mean_elapsed = _mean_or_none(elapsed_values)

        global_summary["by_budget"][time_budget] = {
            "label": label,
            "mean_objective": mean_objective,
            "best_objective": best_objective,
            "worst_objective": worst_objective,
            "mean_elapsed_s": mean_elapsed,
        }

        log_fn(
            f"  {label}. budget={time_budget:.2f}s | "
            f"mean_obj={_fmt_optional_float(mean_objective, 6)} | "
            f"best_obj={_fmt_optional_float(best_objective, 6)} | "
            f"worst_obj={_fmt_optional_float(worst_objective, 6)} | "
            f"mean_time={_fmt_optional_float(mean_elapsed, 4)}s"
        )

    log_fn(EVAL_BIG_SEPARATED)

    return {
        "results": results,
        "summary": global_summary,
    }


def evaluate_solver_cs2sar_with_tsp(
    iterations: int = EVAL_CS2SAR_ITERATIONS,
    instances_fnames_to_evaluate: List[str] = EVAL_INST_FNAMES,
    shuffle: bool = True,
    time_cs2sar: float = 600,
    time_tsp: float = 30,
    verbose_sections: Optional[Iterable[str] | str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Compara CS2SA-R contra TSP+KRP en múltiples instancias.

    Args:
        iterations: Máximo de instancias a evaluar.
        instances_fnames_to_evaluate: Nombres de instancias a evaluar.
        shuffle: Si es ``True``, baraja el orden de instancias.
        time_cs2sar: Presupuesto de tiempo para CS2SA-R.
        time_tsp: Presupuesto de tiempo para TSP y KRP.
        verbose_sections: Secciones de logging reenviadas a
            ``CS2SARSolver.solve()``.
        log_fn: Función de logging. Si es ``None``, usa ``logger.info``.

    Returns:
        Diccionario con resultados por instancia y resumen global.
    """
    # Imports locales: TSP, packing y validación solo se necesitan en este
    # benchmark comparativo.
    from src.ttp_packages.evaluation.config import DEFAULT_N_RESTARTS
    from src.ttp_packages.evaluation.fixed_tour import validate_tour
    from src.ttp_packages.optimization.classical.packing.api import (
        solve_pack_for_fixed_tour,
    )
    from src.ttp_packages.optimization.classical.tsp.tsp_api import (
        solve_tsp_with_ortools,
    )

    if log_fn is None:
        log_fn = logger.info

    instances_fnames = list(instances_fnames_to_evaluate)
    if shuffle:
        random.shuffle(instances_fnames)

    results: List[Dict[str, Any]] = []

    for instance_index, file_name in enumerate(instances_fnames):
        if instance_index >= iterations:
            break

        log_fn(EVAL_BIG_SEPARATED)
        log_fn(f"\nInstancia: {file_name}")

        instance = load_instance(fname=file_name)

        log_fn(EVAL_SEPARATED)
        log_fn("Soluciones:")

        start_cs2sar = perf_counter()
        solver = CS2SARSolver(instance, seed=12345)
        solution_cs2sar = solver.solve(
            time_budget_s=time_cs2sar,
            verbose_sections=verbose_sections,
            log_fn=log_fn,
        )
        validate_tour(instance, solution_cs2sar.tour, start_city=0)
        elapsed_cs2sar = perf_counter() - start_cs2sar

        start_tsp = perf_counter()
        raw_tsp_tour = solve_tsp_with_ortools(
            instance,
            time_limit=time_tsp,
            verbose=False,
        )
        validate_tour(instance, raw_tsp_tour, start_city=0)

        solution_tsp = solve_pack_for_fixed_tour(
            instance,
            raw_tsp_tour,
            time_budget_s=time_tsp,
            n_restarts=DEFAULT_N_RESTARTS,
            seed=12345,
            log_fn=log_fn,
        )
        elapsed_tsp = perf_counter() - start_tsp

        log_fn(f"    CS2SA-R objective: {solution_cs2sar.objective}")
        log_fn(f"    TSP+KRP objective: {solution_tsp.objective}")

        results.append(
            {
                "instance": file_name,
                "cs2sar_objective": float(solution_cs2sar.objective),
                "elapsed_cs2sar": float(elapsed_cs2sar),
                "tsp_objective": float(solution_tsp.objective),
                "elapsed_tsp": float(elapsed_tsp),
            }
        )

    global_summary = {
        "n_instances": len(results),
    }

    log_fn(EVAL_BIG_SEPARATED)
    log_fn("\nRESUMEN GLOBAL")
    log_fn(EVAL_SEPARATED)
    log_fn(f"Instancias evaluadas: {len(results)}")
    log_fn(EVAL_BIG_SEPARATED)

    return {
        "results": results,
        "summary": global_summary,
    }


def benchmark_cs2sar_vs_tsp_work(
    iterations: int = EVAL_CS2SAR_ITERATIONS,
    instances_fnames_to_evaluate: List[str] = EVAL_INST_FNAMES,
    shuffle: bool = True,
    time_cs2sar: float = 600,
    time_tsp: float = 30,
    verbose_sections: Optional[Iterable[str] | str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Ejecuta el workflow de benchmark CS2SA-R vs TSP+KRP.

    Args:
        iterations: Máximo de instancias a evaluar.
        instances_fnames_to_evaluate: Nombres de instancias a evaluar.
        shuffle: Si es ``True``, baraja el orden de instancias.
        time_cs2sar: Presupuesto de tiempo para CS2SA-R.
        time_tsp: Presupuesto de tiempo para TSP y KRP.
        verbose_sections: Secciones de logging reenviadas a CS2SA-R.
        log_fn: Función de logging. Si es ``None``, usa ``logger.info``.

    Returns:
        Diccionario con resultados y resumen del benchmark.
    """
    if log_fn is None:
        log_fn = logger.info

    return evaluate_solver_cs2sar_with_tsp(
        iterations=iterations,
        instances_fnames_to_evaluate=instances_fnames_to_evaluate,
        shuffle=shuffle,
        time_cs2sar=time_cs2sar,
        time_tsp=time_tsp,
        verbose_sections=verbose_sections,
        log_fn=log_fn,
    )


def evaluate_solver_cs2sar_work(
    iterations: int = EVAL_CS2SAR_ITERATIONS,
    instances_fnames_to_evaluate: List[str] = EVAL_INST_FNAMES,
    time_solutions: List[int] = EVAL_LIST_TIME_TO_SOL,
    shuffle: bool = True,
    verbose_sections: Optional[Iterable[str] | str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evalúa CS2SA-R y agrega métricas de overtime por presupuesto.

    Args:
        iterations: Máximo de instancias a evaluar.
        instances_fnames_to_evaluate: Nombres de instancias a evaluar.
        time_solutions: Presupuestos de tiempo por corrida.
        shuffle: Si es ``True``, baraja el orden de instancias.
        verbose_sections: Secciones de logging reenviadas a CS2SA-R.
        log_fn: Función de logging. Si es ``None``, usa ``logger.info``.

    Returns:
        Diccionario con resultados enriquecidos y resumen por presupuesto.
    """
    if log_fn is None:
        log_fn = logger.info

    out = evaluate_solver_cs2sar(
        iterations=iterations,
        instances_fnames_to_evaluate=instances_fnames_to_evaluate,
        time_solutions=time_solutions,
        shuffle=shuffle,
        verbose_sections=verbose_sections,
        log_fn=log_fn,
    )

    enriched_results: List[Dict[str, Any]] = []
    enriched_summary: Dict[str, Any] = {
        "n_instances": int(out["summary"]["n_instances"]),
        "by_budget": {},
    }

    by_budget_overtime: Dict[float, List[float]] = {
        float(time_budget): [] for time_budget in time_solutions
    }
    by_budget_ratio: Dict[float, List[float]] = {
        float(time_budget): [] for time_budget in time_solutions
    }

    for row in out["results"]:
        row_out = {
            "instance": row["instance"],
            "solutions": {},
        }

        for time_budget, solution_info in row["solutions"].items():
            budget = float(time_budget)
            elapsed = float(solution_info["elapsed_s"])
            overtime = float(elapsed - budget)
            ratio = float(elapsed / budget) if budget > 0.0 else float("inf")

            # Conserva los datos base y agrega métricas de cumplimiento de budget.
            row_out["solutions"][time_budget] = {
                **solution_info,
                "budget_s": budget,
                "overtime_s": overtime,
                "ratio_elapsed_over_budget": ratio,
            }

            by_budget_overtime[budget].append(overtime)
            by_budget_ratio[budget].append(ratio)

        enriched_results.append(row_out)

    for time_budget in time_solutions:
        budget = float(time_budget)
        base_summary = out["summary"]["by_budget"][time_budget]

        overtime_values = by_budget_overtime[budget]
        ratio_values = by_budget_ratio[budget]

        mean_overtime = _mean_or_none(overtime_values)
        max_overtime = float(max(overtime_values)) if overtime_values else None
        mean_ratio = _mean_or_none(ratio_values)
        max_ratio = float(max(ratio_values)) if ratio_values else None

        enriched_summary["by_budget"][time_budget] = {
            **base_summary,
            "mean_overtime_s": mean_overtime,
            "max_overtime_s": max_overtime,
            "mean_ratio_elapsed_over_budget": mean_ratio,
            "max_ratio_elapsed_over_budget": max_ratio,
        }

        log_fn(
            f"  budget={budget:.2f}s | "
            f"mean_overtime={_fmt_optional_float(mean_overtime, 4)}s | "
            f"max_overtime={_fmt_optional_float(max_overtime, 4)}s | "
            f"mean_ratio={_fmt_optional_float(mean_ratio, 4)}x | "
            f"max_ratio={_fmt_optional_float(max_ratio, 4)}x"
        )

    return {
        "results": enriched_results,
        "summary": enriched_summary,
    }