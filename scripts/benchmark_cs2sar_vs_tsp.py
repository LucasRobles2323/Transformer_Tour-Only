#!/usr/bin/python
# -*- coding: utf-8 -*-

# scripts/benchmark_cs2sar_vs_tsp.py

"""CLI para comparar CS2SA-R contra TSP+KRP.

Este script lee una configuración JSON, resuelve el benchmark desde la capa
``application`` y muestra un resumen final.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import _load_json, _resolve_log_fn

from src.ttp_packages.application.benchmark_solvers import (
    benchmark_cs2sar_vs_tsp_work,
)


def run(config_path: str | Path) -> None:
    """Ejecuta el benchmark CS2SA-R vs TSP+KRP.

    Args:
        config_path: Ruta al archivo JSON de configuración.
    """
    cfg = _load_json(config_path)
    log_fn = _resolve_log_fn(cfg.get("log_fn", "print"))

    out = benchmark_cs2sar_vs_tsp_work(
        iterations=int(cfg.get("iterations", 2)),
        instances_fnames_to_evaluate=list(cfg.get("instances_fnames_to_evaluate", [])),
        shuffle=bool(cfg.get("shuffle", True)),
        time_cs2sar=float(cfg.get("time_cs2sar", 600.0)),
        time_tsp=float(cfg.get("time_tsp", 30.0)),
        verbose_sections=cfg.get("verbose_sections", None),
        log_fn=log_fn,
    )

    if log_fn is not None:
        log_fn("\n[OK] Benchmark finalizado.")
        log_fn(f"Instancias corridas: {len(out['results'])}")
        log_fn(f"Summary: {out['summary']}")


def main(argv: list[str] | None = None) -> None:
    """Parsea argumentos CLI y ejecuta el benchmark.

    Args:
        argv: Argumentos opcionales para testing. Si es ``None``, usa
            ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Compara CS2SA-R vs TSP+KRP sobre instancias TTP.",
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation/benchmark_cs2sar_vs_tsp.json",
        help="Ruta al JSON de configuración.",
    )

    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()