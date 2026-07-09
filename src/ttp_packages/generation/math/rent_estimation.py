#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/generation/math/rent_estimation.py

from __future__ import annotations

from time import perf_counter
from typing import Callable, List, Optional

from src.ttp_packages.domain.instance import TTPInstance
from src.ttp_packages.domain.objective import calculate_time
from src.ttp_packages.infrastructure.logging import setup_logger

from .config import (
    DEFAULT_2OPT_ITERS,
    DEFAULT_EPSILON,
    ERR_CRITICAL_TIME,
    KNAPSACK_THRESHOLD_CAPACITY,
    KNAPSACK_THRESHOLD_ITEMS,
)
from .knapsack_proxy import (
    solve_knapsack_dp,
    solve_knapsack_fptas,
    solve_knapsack_scip_isolated,
)
from .tsp_proxy import build_tour_nn_2opt

# Inicialización del logger para este módulo específico
logger = setup_logger(__name__)


def _build_reference_tour(
    inst: TTPInstance,
    *,
    tour_mode: str = "nn_2opt",
    tsp_time_limit_s: float = 2.0,
    two_opt_iters: int = DEFAULT_2OPT_ITERS,
    verbose: bool = False,
) -> List[int]:
    """Construye un tour proxy para estimar el factor de renta.

    Args:
        inst: Instancia TTP.
        tour_mode: Método de construcción del tour. Valores: ``"nn_2opt"`` u
            ``"ortools"``.
        tsp_time_limit_s: Límite de tiempo para OR-Tools.
        two_opt_iters: Iteraciones máximas para 2-opt.
        verbose: Si es True, permite logging detallado en métodos internos.

    Returns:
        Tour proxy como lista de IDs de ciudades.

    Raises:
        ValueError: Si ``tour_mode`` no es válido.
    """
    mode = str(tour_mode).strip().lower()

    if mode == "nn_2opt":
        return build_tour_nn_2opt(
            inst,
            two_opt_iters=max(0, int(two_opt_iters)),
            verbose=verbose,
        )

    if mode == "ortools":
        from src.ttp_packages.optimization.classical.tsp.tsp_api import (
            solve_tsp_with_ortools,
        )

        # OR-Tools trabaja con límite entero en varios backends.
        limit_s = max(1, int(round(float(tsp_time_limit_s))))
        return solve_tsp_with_ortools(
            inst,
            time_limit=limit_s,
            verbose=verbose,
        )

    raise ValueError(
        f"tour_mode inválido: {tour_mode}. "
        "Valores válidos: 'nn_2opt', 'ortools'."
    )


def compute_rent_factor(
    inst: TTPInstance,
    tour: Optional[List[int]] = None,
    *,
    eps: float = DEFAULT_EPSILON,
    tour_mode: str = "nn_2opt",
    tsp_time_limit_s: float = 2.0,
    two_opt_iters: int = DEFAULT_2OPT_ITERS,
    knapsack_mode: str = "auto",
    knapsack_time_limit_s: float = 2.0,
    knapsack_gap: float = 0.05,
    knapsack_hide_output: bool = True,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> float:
    """Estima el factor de renta ``R`` para una instancia TTP.

    La estimación usa un tour proxy y una solución proxy de mochila. Luego calcula
    el tiempo físico del tour con el packing seleccionado y define ``R`` como:

    ``R = profit_proxy / time_proxy``

    Args:
        inst: Instancia TTP.
        tour: Tour opcional. Si es ``None``, se construye uno con ``tour_mode``.
        eps: Tolerancia para FPTAS.
        tour_mode: Método para construir el tour proxy.
        tsp_time_limit_s: Límite de tiempo para el solver TSP proxy.
        two_opt_iters: Iteraciones máximas de 2-opt.
        knapsack_mode: Método de mochila: ``"auto"``, ``"dp"``, ``"fptas"`` o
            ``"scip"``.
        knapsack_time_limit_s: Límite de tiempo para SCIP.
        knapsack_gap: Gap relativo para SCIP.
        knapsack_hide_output: Si es True, oculta salida de SCIP.
        verbose: Si es True, registra tiempos y decisiones.
        log_fn: Función opcional de logging.

    Returns:
        Factor de renta estimado.

    Raises:
        ValueError: Si ``knapsack_mode`` es inválido.
        RuntimeError: Si el tiempo físico calculado no es positivo.
    """

    if log_fn is None:
        log_fn = logger.info

    t0_total = perf_counter()

    # 1) Tour proxy
    t0_tour = perf_counter()
    if tour is None:
        if verbose:
            log_fn(
                f"[RENT] Tour no proporcionado. Generando proxy con mode={tour_mode} | "
                f"tsp_time_limit_s={float(tsp_time_limit_s):.2f} | "
                f"two_opt_iters={int(two_opt_iters)}"
            )
        tour = _build_reference_tour(
            inst,
            tour_mode=tour_mode,
            tsp_time_limit_s=tsp_time_limit_s,
            two_opt_iters=two_opt_iters,
            verbose=verbose,
        )
    t_tour = perf_counter() - t0_tour

    if verbose:
        log_fn(
            f"[RENT] TOUR | mode={tour_mode} | "
            f"time={t_tour:.3f}s | len={len(tour)}"
        )

    # 2) Mochila proxy
    t0_knap = perf_counter()
    kmode = str(knapsack_mode).strip().lower()

    if kmode == "scip":
        if verbose:
            log_fn(
                f"[RENT] KNAP | method=SCIP | "
                f"time_limit={float(knapsack_time_limit_s):.2f}s | "
                f"gap={float(knapsack_gap):.5f} | "
                f"n_items={len(inst.items)} | capacity={float(inst.capacity):.1f}"
            )
        z_opt, picked_set = solve_knapsack_scip_isolated(
            inst,
            time_limit_s=float(knapsack_time_limit_s),
            rel_gap=float(knapsack_gap),
            hide_output=bool(knapsack_hide_output),
            verbose=verbose,
            log_fn=log_fn,
        )
        knap_method = "SCIP"

    elif kmode == "dp":
        if verbose:
            log_fn(
                f"[RENT] KNAP | method=DP | "
                f"n_items={len(inst.items)} | capacity={float(inst.capacity):.1f}"
            )
        z_opt, picked_set = solve_knapsack_dp(inst)
        knap_method = "DP"

    elif kmode == "fptas":
        if verbose:
            log_fn(
                f"[RENT] KNAP | method=FPTAS | "
                f"eps={float(eps):.5f} | "
                f"n_items={len(inst.items)} | capacity={float(inst.capacity):.1f}"
            )
        z_opt, picked_set = solve_knapsack_fptas(inst, eps=eps)
        knap_method = "FPTAS"

    elif kmode == "auto":
        use_fptas = (
            len(inst.items) > KNAPSACK_THRESHOLD_ITEMS
            or inst.capacity > KNAPSACK_THRESHOLD_CAPACITY
        )

        if use_fptas:
            if verbose:
                log_fn(
                    f"[RENT] KNAP | method=FPTAS(auto) | "
                    f"eps={float(eps):.5f} | "
                    f"n_items={len(inst.items)} | capacity={float(inst.capacity):.1f}"
                )
            z_opt, picked_set = solve_knapsack_fptas(inst, eps=eps)
            knap_method = "FPTAS"
        else:
            if verbose:
                log_fn(
                    f"[RENT] KNAP | method=DP(auto) | "
                    f"n_items={len(inst.items)} | capacity={float(inst.capacity):.1f}"
                )
            z_opt, picked_set = solve_knapsack_dp(inst)
            knap_method = "DP"

    else:
        raise ValueError(
            f"knapsack_mode inválido: {knapsack_mode}. "
            f"Valores válidos: 'auto', 'dp', 'fptas', 'scip'."
        )
    
    t_knap = perf_counter() - t0_knap
    
    # 3) Packing binario
    t0_pack = perf_counter()
    id_to_idx = {it.id: idx for idx, it in enumerate(inst.items)}
    binary_packing = [0] * len(inst.items)
    for item_id in picked_set:
        idx = id_to_idx.get(item_id)
        if idx is not None:
            binary_packing[idx] = 1
    t_pack = perf_counter() - t0_pack

    # 4) Tiempo físico
    t0_time = perf_counter()
    time_total = calculate_time(inst, tour, binary_packing)
    t_time = perf_counter() - t0_time

    if time_total <= 0:
        logger.error(ERR_CRITICAL_TIME)
        raise RuntimeError(ERR_CRITICAL_TIME)
    
    if z_opt <= 0:
        raise RuntimeError("No se pudo estimar R porque el profit proxy es <= 0.")

    rent = z_opt / time_total
    t_total = perf_counter() - t0_total

    if verbose:
        log_fn(
            f"[RENT] TIME_EVAL | time={t_time:.3f}s | "
            f"time_total={float(time_total):.6f}"
        )
        log_fn(
            f"[RENT] BUILD_PACK | time={t_pack:.3f}s"
        )
        log_fn(
            f"[RENT] END | total={t_total:.3f}s | "
            f"tour={t_tour:.3f}s | knap={t_knap:.3f}s | "
            f"pack={t_pack:.3f}s | time_eval={t_time:.3f}s | "
            f"R={float(rent):.6f}"
        )

    return rent
