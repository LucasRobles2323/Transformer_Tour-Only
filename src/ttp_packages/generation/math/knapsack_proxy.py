#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/generation/math/knapsack_proxy.py

from __future__ import annotations

from typing import Callable, Optional, Set, Tuple

from src.ttp_packages.domain.instance import TTPInstance
from src.ttp_packages.infrastructure.logging import setup_logger

from .config import DEFAULT_EPSILON

# Inicialización del logger para este módulo específico
logger = setup_logger(__name__)

def solve_knapsack_scip(
    inst: TTPInstance,
    time_limit_s: float = 2.0,
    rel_gap: float = 0.05,
    hide_output: bool = True,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[float, Set[int]]:
    """Resuelve la mochila 0/1 usando PySCIPOpt.

    Args:
        inst: Instancia TTP con ítems y capacidad.
        time_limit_s: Límite de tiempo del solver en segundos.
        rel_gap: Gap relativo permitido.
        hide_output: Si es True, oculta la salida de SCIP.
        verbose: Si es True, registra métricas del proceso.
        log_fn: Función opcional de logging.

    Returns:
        Tupla ``(profit, picked)`` con el beneficio total y los IDs elegidos.

    Raises:
        RuntimeError: Si PySCIPOpt no está instalado.
    """
    if log_fn is None:
        log_fn = logger.info

    try:
        from pyscipopt import Model, quicksum
    except ImportError as e:
        raise RuntimeError(
            "PySCIPOpt no está instalado. "
            "Instálalo o usa compute_R_knapsack_mode='fptas'/'dp'."
        ) from e

    valid_items = [it for it in inst.items if it.profit > 0 and it.weight > 0]
    if not valid_items or inst.capacity <= 0:
        return 0.0, set()

    model = Model("rent_knapsack_proxy")

    if hide_output:
        model.hideOutput()

    # Variables binarias
    x = {
        it.id: model.addVar(vtype="B", name=f"x_{it.id}")
        for it in valid_items
    }

    # Restricción de capacidad
    model.addCons(
        quicksum(float(it.weight) * x[it.id] for it in valid_items) <= float(inst.capacity),
        name="capacity",
    )

    # Objetivo: maximizar profit
    model.setObjective(
        quicksum(float(it.profit) * x[it.id] for it in valid_items),
        sense="maximize",
    )

    # Límites
    model.setParam("limits/time", float(time_limit_s))
    model.setParam("limits/gap", float(rel_gap))

    if verbose:
        log_fn(
            f"[SCIP-KNAP] start | time_limit={float(time_limit_s):.2f}s | "
            f"gap={float(rel_gap):.5f} | n_items={len(valid_items)} | "
            f"capacity={float(inst.capacity):.1f}"
        )
    
    try:
        model.optimize()
    except KeyboardInterrupt:
        if verbose:
            log_fn("[SCIP-KNAP] KeyboardInterrupt recibido durante optimize()")
        raise

    if verbose:
        log_fn("[SCIP-KNAP] optimize() returned")

    status = str(model.getStatus())
    best_sol = model.getBestSol()

    if best_sol is None:
        if verbose:
            log_fn(
                f"[SCIP-KNAP] status={status} | "
                f"time={model.getSolvingTime():.3f}s | "
                f"gap={model.getGap():.6f} | sin solución factible"
            )
        return 0.0, set()

    picked: Set[int] = set()
    total_profit = 0.0

    for it in valid_items:
        val = model.getSolVal(best_sol, x[it.id])
        if val > 0.5:
            picked.add(it.id)
            total_profit += float(it.profit)

    if verbose:
        log_fn(
            f"[SCIP-KNAP] status={status} | "
            f"time={model.getSolvingTime():.3f}s | "
            f"gap={model.getGap():.6f} | "
            f"picked={len(picked)} | profit={total_profit:.3f}"
        )

    return total_profit, picked

def _solve_knapsack_scip_worker(
    conn,
    inst,
    time_limit_s: float,
    rel_gap: float,
    hide_output: bool,
    verbose: bool,
):
    """
    Worker aislado para ejecutar SCIP en un proceso hijo.
    Envía el resultado por Pipe y termina.
    """
    try:
        profit, picked = solve_knapsack_scip(
            inst,
            time_limit_s=time_limit_s,
            rel_gap=rel_gap,
            hide_output=hide_output,
            verbose=verbose,
            log_fn=None,   # evita ruido desde el hijo
        )
        conn.send(("ok", float(profit), list(sorted(picked))))
    except BaseException as e:
        conn.send(("err", repr(e)))
    finally:
        conn.close()

def solve_knapsack_scip_isolated(
    inst: TTPInstance,
    time_limit_s: float = 2.0,
    rel_gap: float = 0.05,
    hide_output: bool = True,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
    poll_interval_s: float = 0.10,
) -> Tuple[float, Set[int]]:
    """
    Ejecuta SCIP en un subproceso aislado.
    Si el padre recibe Ctrl+C, puede matar el hijo sin quedar pegado
    dentro de model.optimize().
    """
    import multiprocessing as mp
    import time

    if log_fn is None:
        log_fn = logger.info

    ctx = mp.get_context("spawn")   # importante en Windows
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(
        target=_solve_knapsack_scip_worker,
        args=(
            child_conn,
            inst,
            float(time_limit_s),
            float(rel_gap),
            bool(hide_output),
            bool(verbose),
        ),
        daemon=True,
    )

    start = time.perf_counter()
    proc.start()
    child_conn.close()

    try:
        while True:
            if parent_conn.poll(poll_interval_s):
                msg = parent_conn.recv()
                tag = msg[0]

                if tag == "ok":
                    profit = float(msg[1])
                    picked = set(map(int, msg[2]))
                    if verbose:
                        elapsed = time.perf_counter() - start
                        log_fn(
                            f"[SCIP-KNAP-ISO] done | elapsed={elapsed:.3f}s | "
                            f"picked={len(picked)} | profit={profit:.3f}"
                        )
                    proc.join(timeout=1.0)
                    return profit, picked

                if tag == "err":
                    raise RuntimeError(f"SCIP worker falló: {msg[1]}")

            if not proc.is_alive():
                # murió sin mandar resultado
                raise RuntimeError("SCIP worker terminó sin devolver resultado.")

    except KeyboardInterrupt:
        if verbose:
            log_fn("[SCIP-KNAP-ISO] Ctrl+C recibido. Terminando subproceso SCIP...")

        proc.terminate()
        proc.join(timeout=2.0)

        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)

        raise

    finally:
        parent_conn.close()
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)

def solve_knapsack_dp(inst: TTPInstance) -> Tuple[float, Set[int]]:
    """Resuelve el problema de la mochila (0/1) de forma exacta usando DP.

    Args:
        inst: La instancia del problema TTP que contiene items y capacidad.

    Returns:
        A tuple containing:
            - float: El beneficio máximo obtenido.
            - Set[int]: Conjunto de IDs de los ítems seleccionados.
    """
    w_max = int(inst.capacity)
    items = inst.items
    n_items = len(items)

    if w_max <= 0 or n_items == 0:
        logger.warning("Instancia vacía o capacidad nula detectada.")
        return 0.0, set()

    dp = [0.0] * (w_max + 1)
    choose = [[False] * (w_max + 1) for _ in range(n_items)]

    for i, it in enumerate(items):
        w_i, p_i = int(it.weight), it.profit
        if w_i > w_max or p_i <= 0:
            continue

        for w in range(w_max, w_i - 1, -1):
            cand = dp[w - w_i] + p_i
            if cand > dp[w]:
                dp[w] = cand
                choose[i][w] = True

    w_best = max(range(w_max + 1), key=lambda w: dp[w])
    picked, curr_w = set(), w_best

    for i in reversed(range(n_items)):
        if choose[i][curr_w]:
            picked.add(items[i].id)
            curr_w -= int(items[i].weight)

    return dp[w_best], picked

def solve_knapsack_fptas(inst: TTPInstance, eps: float = DEFAULT_EPSILON) -> Tuple[float, Set[int]]:
    """Esquema de Aproximación Totalmente Polinomial (FPTAS).

    Escala los beneficios para reducir el espacio de búsqueda, sacrificando
    precisión por velocidad en instancias grandes.

    Args:
        inst: La instancia del problema TTP.
        eps: Factor de error tolerable (0 < eps < 1).

    Returns:
        A tuple containing:
            - float: El beneficio real (no escalado) de los ítems seleccionados.
            - Set[int]: Conjunto de IDs de los ítems seleccionados.
    """
    valid_items = [it for it in inst.items if it.profit > 0 and it.weight > 0]
    if not valid_items or inst.capacity <= 0:
        return 0.0, set()

    p_max = max(it.profit for it in valid_items)
    n = len(valid_items)
    k_factor = max(1.0, (eps * p_max / n))

    scaled_items = [(it.id, int(it.weight), int(it.profit // k_factor)) for it in valid_items]
    max_scaled_p = sum(v for _, _, v in scaled_items)

    inf = float('inf')
    dp = [inf] * (max_scaled_p + 1)
    dp[0] = 0
    parent = [(-1, -1)] * (max_scaled_p + 1)

    for it_id, w, v in scaled_items:
        for val in range(max_scaled_p, v - 1, -1):
            if dp[val - v] != inf and (dp[val - v] + w < dp[val]):
                dp[val] = dp[val - v] + w
                parent[val] = (val - v, it_id)

    best_v_scaled = 0
    for v in range(max_scaled_p, -1, -1):
        if dp[v] <= inst.capacity:
            best_v_scaled = v
            break

    picked, curr = set(), best_v_scaled
    while curr > 0 and parent[curr][0] != -1:
        prev_val, it_id = parent[curr]
        picked.add(it_id)
        curr = prev_val

    real_profit = sum(it.profit for it in inst.items if it.id in picked)
    return real_profit, picked