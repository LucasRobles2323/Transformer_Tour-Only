#!/usr/bin/python
# -*- coding: utf-8 -*-

# scripts/optuna_tour.py

"""CLI para ejecutar HPO con Optuna sobre el modelo tour-only.

Este script lee una configuración JSON, construye el search config desde la capa
``application`` y ejecuta el workflow de optimización de hiperparámetros.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import _load_json, _resolve_log_fn

from src.ttp_packages.application.config import (
    OptunaModelSearchSpace,
    OptunaSearchConfig,
    OptunaTrainSearchSpace,
    OptunaWorkflowParams,
)

from src.ttp_packages.application.hpo_optuna import run_optuna_tour_from_dataset_work


def run(config_path: str | Path) -> None:
    """Ejecuta un estudio Optuna desde un archivo de configuración.

    Args:
        config_path: Ruta al archivo JSON de configuración.
    """
    cfg = _load_json(config_path)
    log_fn = _resolve_log_fn(cfg.get("log_fn", "print"))

    workflow = OptunaWorkflowParams(**cfg["workflow"])
    model_space = OptunaModelSearchSpace(**cfg["model_space"])
    train_space = OptunaTrainSearchSpace(**cfg["train_space"])

    search_cfg = OptunaSearchConfig(
        workflow=workflow,
        model_space=model_space,
        train_space=train_space,
    )

    run_optuna_tour_from_dataset_work(
        dataset_file=cfg["dataset_file"],
        search_cfg=search_cfg,
        model_id=cfg.get("model_id", None),
        overwrite_model=bool(cfg.get("overwrite_model", True)),
        overwrite_history=bool(cfg.get("overwrite_history", False)),
        export_best_plots=bool(cfg.get("export_best_plots", True)),
        print_best_result=bool(cfg.get("print_best_result", True)),
        print_final_summary=bool(cfg.get("print_final_summary", True)),
        log_fn=log_fn,
    )


def main(argv: list[str] | None = None) -> None:
    """Parsea argumentos CLI y ejecuta Optuna.

    Args:
        argv: Argumentos opcionales para testing. Si es ``None``, usa
            ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Ejecuta un estudio de Optuna para el modelo tour-only.",
    )
    parser.add_argument(
        "--config",
        default="configs/optuna/optuna_tour.json",
        help="Ruta al JSON de configuración.",
    )

    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()