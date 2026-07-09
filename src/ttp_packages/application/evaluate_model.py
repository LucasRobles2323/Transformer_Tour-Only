#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/evaluate_model.py

"""Workflows de evaluación de modelos neuronales tour-only.

Este módulo genera instancias sintéticas, obtiene soluciones de referencia con
CS2SA-R, TSP+KRP y un baseline random, evalúa uno o más decoders neuronales y
guarda resultados comparativos cuando se evalúa desde un checkpoint.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from . import config as cfg
from .load_instance import get_generated_inst
from .solve_instance import solve_with_cs2sar

from src.ttp_packages.domain.solution import TTPSolution
from src.ttp_packages.domain.tour_ops import validate_tour
from src.ttp_packages.evaluation.tour_metrics import (
    tour_distance,
    tour_edge_set,
)
from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.infrastructure.runtime import get_default_device
from src.ttp_packages.infrastructure.storage.compare_io import save_compare_run
from src.ttp_packages.infrastructure.storage.model_in import (
    import_model_from_checkpoint_file,
)
from src.ttp_packages.optimization.classical.packing.api import (
    solve_pack_for_fixed_tour,
)
from src.ttp_packages.optimization.classical.tsp.tsp_api import (
    solve_tsp_with_ortools,
)
from src.ttp_packages.optimization.classical.ttp.random_solver import (
    solve_random_solution,
    solve_random_tour_with_krp,
)
from src.ttp_packages.optimization.neural.inference import solve_ttp_with_model


logger = setup_logger(__name__)


_MODEL_TAG_RE = re.compile(r"(model[0-9A-Za-z]+)", re.IGNORECASE)

_DECODER_SOURCE_ALIASES = {
    "logit": "logits",
    "logits": "logits",
    "prob": "probs",
    "probs": "probs",
    "probit": "probs",
    "probits": "probs",
}

def _safe_float(value: Any, default: float = float("nan")) -> float:
    """Convierte un valor a ``float`` con fallback seguro.

    Args:
        value: Valor a convertir.
        default: Valor retornado cuando ``value`` es ``None`` o no convertible.

    Returns:
        Valor convertido a ``float`` o ``default``.
    """
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _format_nonempty_tour_issues(row: Dict[str, Any]) -> str:
    """Formatea solo problemas no vacíos de validación de tours.

    Args:
        row: Fila de métricas asociada a un decoder.

    Returns:
        Texto compacto con solo los solvers que tienen issues. Si no hay issues,
        retorna una cadena vacía.
    """
    issue_fields = (
        ("model", "pred_tour_issues"),
        ("tsp", "tsp_tour_issues"),
        ("cs2sar", "cs2sar_tour_issues"),
        ("random", "random_tour_issues"),
        ("rand+krp", "rand_krp_tour_issues"),
    )

    parts = []

    for label, key in issue_fields:
        issues = row.get(key)

        if issues:
            parts.append(f"{label}={issues}")

    return "; ".join(parts)

def _decoder_column_prefix(decoder_source: str) -> str:
    """Devuelve el prefijo visible de columnas para un decoder.

    Args:
        decoder_source: Fuente del decoder, normalmente ``"logits"`` o
            ``"probs"``.

    Returns:
        Prefijo usado en columnas de salida. Para logits retorna
        ``"model_log"`` y para probabilidades retorna ``"model_prob"``.
    """
    source = str(decoder_source).lower()

    if source == "logits":
        return "model_log"

    if source in {"probs", "probits"}:
        return "model_prob"

    return f"model_{source}"

def _short_model_tag_from_checkpoint_file(checkpoint_file: str) -> str:
    """Extrae la etiqueta corta ``modelYY`` desde un checkpoint.

    El nombre completo del checkpoint puede incluir hiperparámetros, tokens de
    arquitectura y extensión ``.pt``. Para los artefactos de comparación solo se
    conserva el identificador corto del modelo, por ejemplo ``"model01"``.

    Args:
        checkpoint_file: Nombre o ruta del checkpoint del modelo.

    Returns:
        Etiqueta corta del modelo. Si no se encuentra un token ``modelYY``, usa
        el primer token del stem del archivo como fallback.
    """
    checkpoint_stem = Path(checkpoint_file).stem
    match = _MODEL_TAG_RE.search(checkpoint_stem)

    if match is not None:
        return match.group(1).lower()

    fallback = checkpoint_stem.split("_", 1)[0].strip().lower()
    return fallback or "model"

def _normalize_decoder_sources(
    decoder_sources: Optional[Sequence[str]],
) -> List[str]:
    """Normaliza la lista de fuentes de decoding a evaluar.

    La etiqueta pública usada en reportes es ``"probs"`` para la evaluación
    basada en probabilidades post-Sinkhorn y ``"logits"`` para la evaluación
    basada en logits. Por compatibilidad, también acepta ``"probits"`` y lo
    normaliza a ``"probs"``.

    Args:
        decoder_sources: Secuencia opcional con ``"logits"``, ``"probs"`` o
            alias compatibles como ``"probits"``.

    Returns:
        Lista de fuentes únicas, en orden, usando etiquetas visibles para
        reporte.

    Raises:
        ValueError: Si se solicita una fuente no soportada.
    """
    if decoder_sources is None:
        decoder_sources = ["logits"]

    normalized: List[str] = []

    for raw_source in decoder_sources:
        source_key = str(raw_source).strip().lower()
        source = _DECODER_SOURCE_ALIASES.get(source_key)

        if source is None:
            raise ValueError(
                "decoder_sources solo admite 'logits', 'probs' o 'probits'. "
                f"Valor recibido: {raw_source!r}"
            )

        if source not in normalized:
            normalized.append(source)

    if not normalized:
        normalized.append("logits")

    return normalized


def _summarize_rows_for_source(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Resume métricas de evaluación para un conjunto de filas.

    Args:
        rows: Resultados detallados de evaluación.

    Returns:
        Diccionario con promedios de objetivo, tiempo, profit y métricas
        estructurales.
    """
    if not rows:
        return {}

    valid_rows = [row for row in rows if row.get("all_tours_valid", False)]
    validity_rate = len(valid_rows) / len(rows)

    if not valid_rows:
        return {
            "n_rows": float(len(rows)),
            "validity_rate": float(validity_rate),
        }

    model_objs = [_safe_float(row.get("model_obj")) for row in valid_rows]
    model_times = [_safe_float(row.get("model_time")) for row in valid_rows]
    model_profits = [_safe_float(row.get("model_profit")) for row in valid_rows]

    tsp_objs = [_safe_float(row.get("tsp_obj")) for row in valid_rows]
    tsp_times = [_safe_float(row.get("tsp_time")) for row in valid_rows]
    tsp_profits = [_safe_float(row.get("tsp_profit")) for row in valid_rows]

    cs2sar_objs = [_safe_float(row.get("cs2sar_obj")) for row in valid_rows]
    cs2sar_times = [_safe_float(row.get("cs2sar_time")) for row in valid_rows]
    cs2sar_profits = [_safe_float(row.get("cs2sar_profit")) for row in valid_rows]

    random_objs = [_safe_float(row.get("random_obj")) for row in valid_rows]
    random_times = [_safe_float(row.get("random_time")) for row in valid_rows]
    random_profits = [_safe_float(row.get("random_profit")) for row in valid_rows]

    rand_krp_objs = [_safe_float(row.get("rand_krp_obj")) for row in valid_rows]
    rand_krp_times = [_safe_float(row.get("rand_krp_time")) for row in valid_rows]
    rand_krp_profits = [
        _safe_float(row.get("rand_krp_profit"))
        for row in valid_rows
    ]

    edge_overlaps = [
        _safe_float(row.get("edge_overlap_cs2sar"))
        for row in valid_rows
    ]
    dist_penalties = [
        _safe_float(row.get("tsp_distance_penalty"))
        for row in valid_rows
    ]

    return {
        "n_rows": float(len(rows)),
        "validity_rate": float(validity_rate),

        "avg_model_obj": float(mean(model_objs)),
        "avg_model_time": float(mean(model_times)),
        "avg_model_profit": float(mean(model_profits)),

        "avg_tsp_obj": float(mean(tsp_objs)),
        "avg_tsp_time": float(mean(tsp_times)),
        "avg_tsp_profit": float(mean(tsp_profits)),

        "avg_cs2sar_obj": float(mean(cs2sar_objs)),
        "avg_cs2sar_time": float(mean(cs2sar_times)),
        "avg_cs2sar_profit": float(mean(cs2sar_profits)),

        "avg_random_obj": float(mean(random_objs)),
        "avg_random_time": float(mean(random_times)),
        "avg_random_profit": float(mean(random_profits)),

        "avg_rand_krp_obj": float(mean(rand_krp_objs)),
        "avg_rand_krp_time": float(mean(rand_krp_times)),
        "avg_rand_krp_profit": float(mean(rand_krp_profits)),

        "avg_edge_overlap_cs2sar": float(mean(edge_overlaps)),
        "avg_tsp_distance_penalty": float(mean(dist_penalties)),
    }


def _summarize_evaluation_results(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resume métricas globales y por decoder.

    Args:
        rows: Resultados detallados por instancia y decoder.

    Returns:
        Diccionario con resumen global y resumen separado por ``decoder_source``.
    """
    if not rows:
        return {}

    decoder_sources = sorted(
        {
            str(row.get("decoder_source", "logits"))
            for row in rows
        }
    )

    by_decoder: Dict[str, Dict[str, float]] = {}

    for decoder_source in decoder_sources:
        source_rows = [
            row
            for row in rows
            if str(row.get("decoder_source", "logits")) == decoder_source
        ]
        by_decoder[decoder_source] = _summarize_rows_for_source(source_rows)

    summary: Dict[str, Any] = {
        "n_rows": float(len(rows)),
        "n_instances": float(
            len({int(row.get("instance_id", -1)) for row in rows})
        ),
        "decoder_sources": decoder_sources,
        "by_decoder": by_decoder,
    }

    # Compatibilidad con reportes antiguos: si solo hay un decoder, deja también
    # las métricas principales en el nivel superior.
    if len(decoder_sources) == 1:
        summary.update(by_decoder[decoder_sources[0]])

    return summary

def _pivot_decoder_rows_for_comparison(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convierte filas por decoder en filas por instancia.

    La evaluación interna genera una fila por combinación
    ``instance_id`` + ``decoder_source``. Para la tabla comparativa final es más
    legible tener una sola fila por instancia y separar los objetivos del modelo
    en columnas ``model_log_*`` y ``model_prob_*``.

    Si una fuente no fue evaluada, sus columnas quedan como ``NaN``.

    Args:
        rows: Filas detalladas generadas por la evaluación.

    Returns:
        Lista de filas consolidadas por instancia.
    """
    if not rows:
        return []

    by_instance: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        instance_id = int(row.get("instance_id", -1))
        decoder_source = str(row.get("decoder_source", "logits"))
        prefix = _decoder_column_prefix(decoder_source)

        if instance_id not in by_instance:
            by_instance[instance_id] = {
                "instance_id": instance_id,
                "seed": row.get("seed"),

                # Columnas del modelo. Quedan en NaN si el decoder no se evaluó.
                "model_log_obj": float("nan"),
                "model_prob_obj": float("nan"),

                # Baselines compartidos por instancia.
                "random_obj": _safe_float(row.get("random_obj")),
                "rand_krp_obj": _safe_float(row.get("rand_krp_obj")),
                "tsp_obj": _safe_float(row.get("tsp_obj")),
                "cs2sar_obj": _safe_float(row.get("cs2sar_obj")),

                # Métricas estructurales. Se separan por decoder porque dependen
                # del tour producido por el modelo.
                "edge_overlap_cs2sar_log": float("nan"),
                "edge_overlap_cs2sar_prob": float("nan"),
                "tsp_distance_penalty_log": float("nan"),
                "tsp_distance_penalty_prob": float("nan"),

                # Tiempo de viaje TTP, no tiempo de cómputo.
                "model_log_time": float("nan"),
                "model_prob_time": float("nan"),
                "random_time": _safe_float(row.get("random_time")),
                "rand_krp_time": _safe_float(row.get("rand_krp_time")),
                "tsp_time": _safe_float(row.get("tsp_time")),
                "cs2sar_time": _safe_float(row.get("cs2sar_time")),

                # Profit de la solución.
                "model_log_profit": float("nan"),
                "model_prob_profit": float("nan"),
                "random_profit": _safe_float(row.get("random_profit")),
                "rand_krp_profit": _safe_float(row.get("rand_krp_profit")),
                "tsp_profit": _safe_float(row.get("tsp_profit")),
                "cs2sar_profit": _safe_float(row.get("cs2sar_profit")),

                # Validez general.
                "all_tours_valid": bool(row.get("all_tours_valid", False)),
            }

        out_row = by_instance[instance_id]

        # Si una instancia tiene varias filas, la validez final exige que todas
        # las variantes evaluadas hayan producido tours válidos.
        out_row["all_tours_valid"] = bool(
            out_row["all_tours_valid"]
            and bool(row.get("all_tours_valid", False))
        )

        out_row[f"{prefix}_obj"] = _safe_float(row.get("model_obj"))
        out_row[f"{prefix}_time"] = _safe_float(row.get("model_time"))
        out_row[f"{prefix}_profit"] = _safe_float(row.get("model_profit"))

        if prefix == "model_log":
            out_row["edge_overlap_cs2sar_log"] = _safe_float(
                row.get("edge_overlap_cs2sar")
            )
            out_row["tsp_distance_penalty_log"] = _safe_float(
                row.get("tsp_distance_penalty")
            )

        elif prefix == "model_prob":
            out_row["edge_overlap_cs2sar_prob"] = _safe_float(
                row.get("edge_overlap_cs2sar")
            )
            out_row["tsp_distance_penalty_prob"] = _safe_float(
                row.get("tsp_distance_penalty")
            )

    return [
        by_instance[instance_id]
        for instance_id in sorted(by_instance)
    ]

def _solve_tsp_baseline_solution(
    instance: Any,
    *,
    time_budget_s: float,
    n_restarts: int,
    seed: Optional[int],
    log_fn: Optional[Callable[[str], None]],
) -> TTPSolution:
    """Construye y evalúa el baseline TSP+KRP.

    Args:
        instance: Instancia TTP.
        time_budget_s: Presupuesto para TSP y packing.
        n_restarts: Cantidad de reinicios para packing.
        seed: Semilla opcional para packing.
        log_fn: Función opcional de logging.

    Returns:
        Solución TTP obtenida con tour TSP y packing optimizado.
    """
    raw_tsp_tour = solve_tsp_with_ortools(
        instance,
        time_limit=time_budget_s,
        verbose=False,
    )

    return solve_pack_for_fixed_tour(
        instance,
        raw_tsp_tour,
        time_budget_s=time_budget_s,
        n_restarts=n_restarts,
        seed=seed,
        log_fn=log_fn,
    )


def _compare_solutions(
    *,
    instance: Any,
    model_solution: TTPSolution,
    tsp_solution: TTPSolution,
    cs2sar_solution: TTPSolution,
    random_solution: TTPSolution,
    rand_krp_solution: TTPSolution,
    decoder_source: str,
    start_city: int = 0,
) -> Dict[str, Any]:
    """Compara solución neuronal contra TSP+KRP, CS2SA-R, random y rand+krp.

    Args:
        instance: Instancia TTP evaluada.
        model_solution: Solución TTP generada por modelo + packing clásico.
        tsp_solution: Solución TSP+KRP.
        cs2sar_solution: Solución CS2SA-R.
        random_solution: Solución TTP generada por solver random.
        rand_krp_solution: Solución TTP con tour random y packing KRP.
        decoder_source: Fuente usada por el decoder neuronal.
        start_city: Ciudad inicial esperada.

    Returns:
        Diccionario con validez de tours, valores objetivo, tiempos, profits y
        métricas estructurales.
    """
    val_model = validate_tour(
        instance,
        model_solution.tour,
        start_city=start_city,
    )
    val_tsp = validate_tour(
        instance,
        tsp_solution.tour,
        start_city=start_city,
    )
    val_cs2sar = validate_tour(
        instance,
        cs2sar_solution.tour,
        start_city=start_city,
    )
    val_random = validate_tour(
        instance,
        random_solution.tour,
        start_city=start_city,
    )
    val_rand_krp = validate_tour(
        instance,
        rand_krp_solution.tour,
        start_city=start_city,
    )

    all_valid = (
        val_model["is_valid"]
        and val_tsp["is_valid"]
        and val_cs2sar["is_valid"]
        and val_random["is_valid"]
        and val_rand_krp["is_valid"]
    )

    out: Dict[str, Any] = {
        "decoder_source": decoder_source,
        "all_tours_valid": bool(all_valid),
        "pred_tour_issues": val_model["issues"],
        "tsp_tour_issues": val_tsp["issues"],
        "cs2sar_tour_issues": val_cs2sar["issues"],
        "random_tour_issues": val_random["issues"],
        "rand_krp_tour_issues": val_rand_krp["issues"],
    }

    if not all_valid:
        return out

    t_model = val_model["normalized_tour"]
    t_tsp = val_tsp["normalized_tour"]
    t_cs2sar = val_cs2sar["normalized_tour"]

    model_edges = tour_edge_set(t_model)
    cs2sar_edges = tour_edge_set(t_cs2sar)

    # Fracción de aristas del tour CS2SA-R que también aparecen en el tour del modelo.
    edge_overlap = len(model_edges & cs2sar_edges) / max(len(cs2sar_edges), 1)

    dist_model = tour_distance(instance, t_model)
    dist_tsp = tour_distance(instance, t_tsp)

    # Métrica informativa: cuánto más largo es el tour del modelo respecto al TSP.
    # No modifica la función objetivo TTP.
    dist_penalty = (dist_model / dist_tsp) - 1.0 if dist_tsp > 0 else 0.0

    out.update(
        {
            "model_obj": float(model_solution.objective),
            "model_time": float(model_solution.time),
            "model_profit": float(model_solution.profit),

            "tsp_obj": float(tsp_solution.objective),
            "tsp_time": float(tsp_solution.time),
            "tsp_profit": float(tsp_solution.profit),

            "cs2sar_obj": float(cs2sar_solution.objective),
            "cs2sar_time": float(cs2sar_solution.time),
            "cs2sar_profit": float(cs2sar_solution.profit),

            "random_obj": float(random_solution.objective),
            "random_time": float(random_solution.time),
            "random_profit": float(random_solution.profit),

            "rand_krp_obj": float(rand_krp_solution.objective),
            "rand_krp_time": float(rand_krp_solution.time),
            "rand_krp_profit": float(rand_krp_solution.profit),

            "edge_overlap_cs2sar": float(edge_overlap),
            "tsp_distance_penalty": float(dist_penalty),
        }
    )

    return out


def evaluate_tour_model_work(
    model: torch.nn.Module,
    n_instances: int,
    base_seed: Optional[int] = cfg.DEFAULT_SEED,
    device: Optional[Union[str, torch.device]] = None,
    inst_params: cfg.InstanceGeneratorParams = cfg.DEFAULT_INST_PARAMS,
    solver_time_budget_s: float = 60.0,
    eval_time_budget_s: float = cfg.DEFAULT_EVAL_TIME_BUDGET_S,
    eval_n_restarts: int = cfg.DEFAULT_EVAL_N_RESTARTS,
    decoder_sources: Optional[Sequence[str]] = None,
    mask_mode: Optional[str] = None,
    knn_k: Optional[int] = None,
    allow_self: Optional[bool] = None,
    sym: Optional[bool] = None,
    compute_dist_matrix: Optional[bool] = None,
    verbose: bool = cfg.VERBOSE_EVALUATE_WORK_MAIN,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evalúa uno o más decoders de un modelo neuronal.

    Genera instancias sintéticas, obtiene CS2SA-R como referencia, calcula
    TSP+KRP como baseline y evalúa cada ``decoder_source`` solicitado usando
    ``solve_ttp_with_model``.

    Args:
        model: Modelo neuronal entrenado.
        n_instances: Cantidad de instancias sintéticas a evaluar.
        base_seed: Semilla base. Cada instancia usa ``base_seed + i``.
        device: Device de inferencia.
        inst_params: Parámetros de generación de instancias.
        solver_time_budget_s: Presupuesto para CS2SA-R.
        eval_time_budget_s: Presupuesto para packing sobre tours fijos.
        eval_n_restarts: Cantidad de reinicios de packing.
        decoder_sources: Fuentes del decoder: ``["logits"]``, ``["probits"]`` o
            ``["logits", "probits"]``. También acepta ``"probs"`` como alias de
            ``"probits"``.
        mask_mode: Modo de máscara de inferencia. ``None`` usa checkpoint/default.
        knn_k: K de KNN. ``None`` usa checkpoint/default.
        allow_self: Control de self-loops. ``None`` usa checkpoint/default.
        sym: Control de simetría. ``None`` usa checkpoint/default.
        compute_dist_matrix: Control de matriz de distancia. ``None`` usa
            checkpoint/default, y se fuerza a True si el modelo lo requiere.
        verbose: Si es True, registra métricas por instancia.
        log_fn: Función opcional de logging.

    Returns:
        Tupla ``(results, summary)`` con métricas detalladas y resumen.
    """
    if n_instances < 1:
        raise ValueError("n_instances debe ser mayor o igual a 1.")

    if log_fn is None:
        log_fn = logger.info

    resolved_decoder_sources = _normalize_decoder_sources(decoder_sources)

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_default_device()

    run_seed = cfg.resolve_seed(base_seed)

    all_metrics: List[Dict[str, Any]] = []

    for instance_id in range(n_instances):
        instance_seed = run_seed + instance_id

        # Cada instancia fija sus semillas para reproducibilidad local.
        random.seed(instance_seed)
        torch.manual_seed(instance_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(instance_seed)

        inst = get_generated_inst(inst_params=inst_params)

        cs2sar_solution = solve_with_cs2sar(
            inst,
            time_to_solve=solver_time_budget_s,
            log_fn=log_fn,
        )

        tsp_solution = _solve_tsp_baseline_solution(
            inst,
            time_budget_s=eval_time_budget_s,
            n_restarts=eval_n_restarts,
            seed=instance_seed,
            log_fn=log_fn,
        )

        random_solution = solve_random_solution(
            inst,
            seed=instance_seed,
            start_city=0,
        )

        rand_krp_solution = solve_random_tour_with_krp(
            inst,
            time_budget_s=eval_time_budget_s,
            n_restarts=eval_n_restarts,
            seed=instance_seed,
            start_city=0,
            log_fn=log_fn,
        )

        instance_metrics: List[Dict[str, Any]] = []

        for decoder_source in resolved_decoder_sources:
            model_solution = solve_ttp_with_model(
                inst,
                model=model,
                device=device,
                start_city=0,
                decoder_source=decoder_source,
                mask_mode=mask_mode,
                knn_k=knn_k,
                allow_self=allow_self,
                sym=sym,
                compute_dist_matrix=compute_dist_matrix,
                packing_time_budget_s=eval_time_budget_s,
                packing_n_restarts=eval_n_restarts,
                packing_seed=instance_seed,
                log_fn=log_fn,
            )

            metrics = _compare_solutions(
                instance=inst,
                model_solution=model_solution,
                tsp_solution=tsp_solution,
                cs2sar_solution=cs2sar_solution,
                random_solution=random_solution,
                rand_krp_solution=rand_krp_solution,
                decoder_source=decoder_source,
                start_city=0,
            )

            metrics["instance_id"] = instance_id
            metrics["seed"] = instance_seed
            all_metrics.append(metrics)
            instance_metrics.append(metrics)

        if verbose:
            # Consolida logits/probs en una sola fila de log para esta instancia.
            log_row = _pivot_decoder_rows_for_comparison(instance_metrics)[0]
            
            invalid_rows = [
                row
                for row in instance_metrics
                if not bool(row.get("all_tours_valid", False))
            ]

            if invalid_rows:
                warning_parts = []

                for row in invalid_rows:
                    decoder_source = str(row.get("decoder_source", "unknown"))
                    issues_text = _format_nonempty_tour_issues(row)

                    if issues_text:
                        warning_parts.append(f"{decoder_source}: {issues_text}")

                if warning_parts:
                    logger.warning(
                        "[%03d] Tour inválido | %s",
                        instance_id,
                        " | ".join(warning_parts),
                    )

            log_fn(
                f"[{instance_id:02d}] | "
                f"model_log={log_row['model_log_obj']:.2f} | "
                f"model_prob={log_row['model_prob_obj']:.2f} | "
                f"random={log_row['random_obj']:.2f} | "
                f"rand+krp={log_row['rand_krp_obj']:.2f} | "
                f"tsp={log_row['tsp_obj']:.2f} | "
                f"cs2sar={log_row['cs2sar_obj']:.2f} | "
                f"dist_pen_log={log_row['tsp_distance_penalty_log']:.2f} | "
                f"dist_pen_prob={log_row['tsp_distance_penalty_prob']:.2f} | "
                f"edge_log={log_row['edge_overlap_cs2sar_log']:.2f} | "
                f"edge_prob={log_row['edge_overlap_cs2sar_prob']:.2f} | "
                f"valid={log_row['all_tours_valid']}"
            )

    summary = _summarize_evaluation_results(all_metrics)
    table_metrics = _pivot_decoder_rows_for_comparison(all_metrics)

    return table_metrics, summary


def evaluate_tour_model_from_checkpoint_work(
    *,
    checkpoint_file: str,
    n_instances: int,
    run_tag: Optional[str] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    inst_params: cfg.InstanceGeneratorParams = cfg.DEFAULT_INST_PARAMS,
    solver_time_budget_s: float = 60.0,
    eval_time_budget_s: float = cfg.DEFAULT_EVAL_TIME_BUDGET_S,
    eval_n_restarts: int = cfg.DEFAULT_EVAL_N_RESTARTS,
    decoder_sources: Optional[Sequence[str]] = None,
    mask_mode: Optional[str] = None,
    knn_k: Optional[int] = None,
    allow_self: Optional[bool] = None,
    sym: Optional[bool] = None,
    compute_dist_matrix: Optional[bool] = None,
    verbose: bool = cfg.VERBOSE_EVALUATE_WORK_MAIN,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Carga un checkpoint, evalúa el modelo y guarda resultados.

    Args:
        checkpoint_file: Archivo del checkpoint ``.pt``.
        n_instances: Cantidad de instancias sintéticas a evaluar.
        run_tag: Identificador opcional de la corrida.
        config_snapshot: Configuración usada para reproducibilidad del reporte.
        seed: Semilla opcional para la evaluación.
        device: Device de inferencia.
        inst_params: Parámetros de generación de instancias.
        solver_time_budget_s: Presupuesto para CS2SA-R.
        eval_time_budget_s: Presupuesto para packing sobre tours fijos.
        eval_n_restarts: Cantidad de reinicios usados en packing.
        decoder_sources: Fuentes del decoder: ``["logits"]``, ``["probs"]`` o
            ``["logits", "probs"]``.
        mask_mode: Modo de máscara. ``None`` usa checkpoint/default.
        knn_k: K de KNN. ``None`` usa checkpoint/default.
        allow_self: Control de self-loops. ``None`` usa checkpoint/default.
        sym: Control de simetría. ``None`` usa checkpoint/default.
        compute_dist_matrix: Control de matriz de distancia.
        verbose: Si es True, registra métricas por instancia.
        log_fn: Función opcional de logging.

    Returns:
        Diccionario con ``run_tag``, ``summary``, ``results``, ``json_path`` y
        metadata del modelo cargado.
    """
    if log_fn is None:
        log_fn = logger.info

    base_seed = cfg.resolve_seed(seed)

    model, model_meta = import_model_from_checkpoint_file(
        checkpoint_file=checkpoint_file,
        device=device,
        eval_mode=True,
    )

    results, summary = evaluate_tour_model_work(
        model=model,
        n_instances=n_instances,
        base_seed=base_seed,
        device=device,
        inst_params=inst_params,
        solver_time_budget_s=solver_time_budget_s,
        eval_time_budget_s=eval_time_budget_s,
        eval_n_restarts=eval_n_restarts,
        decoder_sources=decoder_sources,
        mask_mode=mask_mode,
        knn_k=knn_k,
        allow_self=allow_self,
        sym=sym,
        compute_dist_matrix=compute_dist_matrix,
        verbose=verbose,
        log_fn=log_fn,
    )

    if run_tag is None:
        run_tag = Path(checkpoint_file).stem

    model_tag = _short_model_tag_from_checkpoint_file(checkpoint_file)

    export_path = save_compare_run(
        run_tag=run_tag,
        config_snapshot=(config_snapshot or {}),
        summary=summary,
        results=results,
        file_name=f"compare{run_tag}_{model_tag}.json",
        table_file_name=f"compare{run_tag}_results.png",
    )

    return {
        "run_tag": run_tag,
        "summary": summary,
        "results": results,
        "json_path": export_path,
        "model_meta": model_meta,
    }


def evaluate_sol_cs2sar_work(
    iterations: int = 2,
    shuffle: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Ejecuta la evaluación auxiliar de CS2SA-R.

    Args:
        iterations: Cantidad de iteraciones por instancia.
        shuffle: Si es True, mezcla el orden de instancias.
        log_fn: Función opcional de logging.
    """
    if log_fn is None:
        log_fn = logger.info

    # Import local: evita cargar benchmark_solvers si solo se evalúa un modelo.
    from src.ttp_packages.application.benchmark_solvers import evaluate_solver_cs2sar

    evaluate_solver_cs2sar(
        iterations=iterations,
        shuffle=shuffle,
        log_fn=log_fn,
    )