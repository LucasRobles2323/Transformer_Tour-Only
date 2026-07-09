#!/usr/bin/python
# -*- coding: utf-8 -*-

# scripts/evaluate_solver_cs2sar.py

"""CLI para evaluar CS2SA-R con distintos presupuestos de tiempo.

Este script lee una configuración JSON, ejecuta el workflow desde la capa
``application`` y muestra un resumen final.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import _load_json, _resolve_log_fn

from src.ttp_packages.application.benchmark_solvers import (
    evaluate_solver_cs2sar_work,
)


def run(config_path: str | Path) -> None:
    """Ejecuta la evaluación de CS2SA-R.

    Args:
        config_path: Ruta al archivo JSON de configuración.
    """
    cfg = _load_json(config_path)
    log_fn = _resolve_log_fn(cfg.get("log_fn", "print"))

    out = evaluate_solver_cs2sar_work(
        iterations=int(cfg.get("iterations", 2)),
        instances_fnames_to_evaluate=list(cfg.get("instances_fnames_to_evaluate", [])),
        time_solutions=list(cfg.get("time_solutions", [30, 60, 300, 420, 600])),
        shuffle=bool(cfg.get("shuffle", True)),
        verbose_sections=cfg.get("verbose_sections", None),
        log_fn=log_fn,
    )

    if log_fn is not None:
        log_fn("\n[OK] Evaluación de CS2SA-R finalizada.")
        log_fn(f"Instancias corridas: {len(out['results'])}")
        log_fn(f"Summary: {out['summary']}")


def main(argv: list[str] | None = None) -> None:
    """Parsea argumentos CLI y ejecuta la evaluación.

    Args:
        argv: Argumentos opcionales para testing. Si es ``None``, usa
            ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Evalúa CS2SA-R con distintos budgets de tiempo.",
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation/evaluate_solver_cs2sar.json",
        help="Ruta al JSON de configuración.",
    )

    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()