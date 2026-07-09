#!/usr/bin/python
# -*- coding: utf-8 -*-

# scripts/gen_dataset.py

"""CLI para generar o extender datasets tensoriales TTP.

Este script lee una configuración JSON, construye los parámetros de generación
desde la capa ``application`` y ejecuta el workflow de generación de dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .utils import _load_json, _resolve_log_fn

from src.ttp_packages.application.config import InstanceGeneratorParams
from src.ttp_packages.application.generate_dataset import generate_tensor_dataset_work


def _build_inst_params(raw: Dict[str, Any]) -> InstanceGeneratorParams:
    """Construye parámetros de generación de instancias desde JSON.

    Args:
        raw: Diccionario leído desde la clave ``inst_params``.

    Returns:
        Objeto ``InstanceGeneratorParams``.
    """
    return InstanceGeneratorParams(**raw)


def run(config_path: str | Path) -> None:
    """Ejecuta el workflow de generación o extensión de dataset.

    Args:
        config_path: Ruta al archivo JSON de configuración.
    """
    cfg = _load_json(config_path)
    log_fn = _resolve_log_fn(cfg.get("log_fn", "print"))

    inst_params = _build_inst_params(cfg.get("inst_params", {}))

    payload = generate_tensor_dataset_work(
        file_name=cfg["file_name"],
        n_new_samples=int(cfg["n_new_samples"]),
        inst_params=inst_params,
        solver_time_budget_s=float(cfg.get("solver_time_budget_s", 60.0)),
        solver_restart_mode=cfg.get("solver_restart_mode", "full"),
        solver_no_improve_patience=int(cfg.get("solver_no_improve_patience", 3)),
        solver_verbose_sections=cfg.get("solver_verbose_sections", None),
        solver_verify_integrity=bool(cfg.get("solver_verify_integrity", False)),
        log_fn=log_fn,
        verbose=bool(cfg.get("verbose", True)),
        verbose_storage=bool(cfg.get("verbose_storage", True)),
        verbose_format=bool(cfg.get("verbose_format", False)),
        seed=cfg.get("seed", None),
    )

    if log_fn is not None:
        log_fn("\n[OK] Dataset generado o extendido.")
        log_fn(f"  n_cities: {payload['n_cities']}")
        log_fn(f"  m_items: {payload['m_items']}")
        log_fn(f"  num_samples: {payload['num_samples']}")


def main(argv: list[str] | None = None) -> None:
    """Parsea argumentos CLI y ejecuta la generación.

    Args:
        argv: Argumentos opcionales para testing. Si es ``None``, usa
            ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Genera o extiende un dataset TTP en formato tensorial.",
    )
    parser.add_argument(
        "--config",
        default="configs/datasets/generate_dataset.json",
        help="Ruta al JSON de configuración.",
    )

    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()