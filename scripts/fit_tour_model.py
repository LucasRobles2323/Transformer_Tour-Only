#!/usr/bin/python
# -*- coding: utf-8 -*-

# scripts/fit_tour_model.py

"""CLI para entrenar un modelo TTP tour-only.

Este script lee una configuración JSON, construye los parámetros de modelo y
entrenamiento desde la capa ``application`` y ejecuta el workflow de training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import _load_json, _resolve_log_fn


from src.ttp_packages.application.config import (
    TTPArchitectureParams,
    TrainingParams,
)

from src.ttp_packages.application.train_model import fit_tour_model_work


def run(config_path: str | Path) -> None:
    """Ejecuta el entrenamiento desde un archivo de configuración.

    Args:
        config_path: Ruta al archivo JSON de configuración.
    """
    cfg = _load_json(config_path)
    log_fn = _resolve_log_fn(cfg.get("log_fn", "print"))

    model_params = TTPArchitectureParams(**cfg["model_params"])
    train_params = TrainingParams(**cfg["train_params"])

    out = fit_tour_model_work(
        dataset_file=cfg["dataset_file"],
        model_id=cfg["model_id"],
        model_params=model_params,
        train_params=train_params,
        export_plots=bool(cfg.get("export_plots", True)),
        dpi=int(cfg.get("dpi", 160)),
        overwrite_model=bool(cfg.get("overwrite_model", True)),
        overwrite_history=bool(cfg.get("overwrite_history", False)),
        log_fn=log_fn,
    )

    if log_fn is not None:
        log_fn("\n[OK] Script completado.")
        log_fn(f"run_tag: {out['run_tag']}")
        log_fn(f"summary: {out['summary']}")
        log_fn(f"export_paths: {out['export_paths']}")
        log_fn(f"plot_paths: {out['plot_paths']}")


def main(argv: list[str] | None = None) -> None:
    """Parsea argumentos CLI y ejecuta el entrenamiento.

    Args:
        argv: Argumentos opcionales para testing. Si es ``None``, usa
            ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Entrena un modelo TTP tour-only desde un dataset guardado.",
    )
    parser.add_argument(
        "--config",
        default="configs/training/fit_tour_model.json",
        help="Ruta al JSON de configuración.",
    )

    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()