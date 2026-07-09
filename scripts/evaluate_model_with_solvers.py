#!/usr/bin/python
# -*- coding: utf-8 -*-

# scripts/evaluate_model_with_solvers.py

"""CLI para evaluar un checkpoint neuronal contra solvers clásicos.

Este script lee una configuración JSON, construye los parámetros de generación
de instancias y ejecuta la evaluación desde la capa ``application``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .utils import _load_json, _resolve_log_fn

from src.ttp_packages.application.config import InstanceGeneratorParams
from src.ttp_packages.application.evaluate_model import (
    evaluate_tour_model_from_checkpoint_work,
)


def _build_inst_params(raw: Dict[str, Any]) -> InstanceGeneratorParams:
    """Construye parámetros de generación desde el bloque JSON.

    Args:
        raw: Diccionario con parámetros de instancia.

    Returns:
        Parámetros de generación de instancias.
    """
    return InstanceGeneratorParams(**raw)


def run(config_path: str | Path) -> None:
    """Ejecuta la evaluación de un checkpoint contra solvers.

    Args:
        config_path: Ruta al archivo JSON de configuración.
    """
    cfg = _load_json(config_path)
    log_fn = _resolve_log_fn(cfg.get("log_fn", "print"))

    inst_params_raw = cfg.get("inst_params", {})
    inst_params = _build_inst_params(inst_params_raw)

    out = evaluate_tour_model_from_checkpoint_work(
        checkpoint_file=cfg["checkpoint_file"],
        n_instances=int(cfg.get("n_instances", 10)),
        run_tag=cfg.get("run_tag", None),
        config_snapshot=cfg,
        seed=cfg.get("seed", None),
        device=cfg.get("device", None),
        inst_params=inst_params,
        solver_time_budget_s=float(cfg.get("solver_time_budget_s", 60.0)),
        eval_time_budget_s=float(cfg.get("eval_time_budget_s", 30.0)),
        eval_n_restarts=int(cfg.get("eval_n_restarts", 3)),

        # Parámetros nuevos de inferencia neuronal.
        decoder_sources=list(cfg.get("decoder_sources", ["logits"])),
        mask_mode=cfg.get("mask_mode", None),
        knn_k=cfg.get("knn_k", None),
        allow_self=cfg.get("allow_self", None),
        sym=cfg.get("sym", None),
        compute_dist_matrix=cfg.get("compute_dist_matrix", None),

        verbose=bool(cfg.get("verbose", True)),
        log_fn=log_fn,
    )

    if log_fn is not None:
        log_fn("\n[OK] Evaluación del modelo finalizada.")
        log_fn(f"run_tag: {out['run_tag']}")
        log_fn(f"summary: {out['summary']}")
        log_fn(f"json_path: {out['json_path']}")


def main(argv: list[str] | None = None) -> None:
    """Parsea argumentos CLI y ejecuta la evaluación.

    Args:
        argv: Argumentos opcionales para testing. Si es ``None``, usa
            ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Evalúa un modelo TTP tour-only desde un checkpoint .pt.",
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation/evaluate_model_with_solvers.json",
        help="Ruta al JSON de configuración.",
    )

    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()