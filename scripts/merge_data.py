#!/usr/bin/python
# -*- coding: utf-8 -*-

# scripts/merge_data.py

"""CLI para unir datasets tensoriales TTP.

Este script lee una configuración JSON mínima con los nombres de los datasets
de entrada y el nombre del dataset combinado. Luego delega la operación al
workflow de aplicación ``merge_tensor_datasets_work``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .utils import _load_json, _resolve_log_fn

from src.ttp_packages.application.merge_dataset import merge_tensor_datasets_work


def _read_input_file_names(cfg: Dict[str, Any]) -> List[str]:
    """Lee y valida los nombres de datasets de entrada desde JSON.

    Args:
        cfg: Configuración cargada desde el archivo JSON.

    Returns:
        Lista de nombres de archivos de dataset.

    Raises:
        ValueError: Si falta ``input_file_names`` o si su contenido no es una
            lista no vacía de strings.
    """
    raw_input_file_names = cfg.get("input_file_names")

    if not isinstance(raw_input_file_names, list) or not raw_input_file_names:
        raise ValueError(
            "La configuración debe incluir 'input_file_names' como lista no vacía."
        )

    input_file_names = [str(file_name).strip() for file_name in raw_input_file_names]

    if any(not file_name for file_name in input_file_names):
        raise ValueError("'input_file_names' contiene nombres vacíos.")

    return input_file_names


def _read_output_file_name(cfg: Dict[str, Any]) -> str:
    """Lee y valida el nombre del dataset combinado desde JSON.

    Args:
        cfg: Configuración cargada desde el archivo JSON.

    Returns:
        Nombre del archivo de salida.

    Raises:
        ValueError: Si falta ``output_file_name`` o si está vacío.
    """
    output_file_name = str(cfg.get("output_file_name", "")).strip()

    if not output_file_name:
        raise ValueError(
            "La configuración debe incluir 'output_file_name' como string no vacío."
        )

    return output_file_name


def run(config_path: str | Path) -> None:
    """Ejecuta el workflow de unión de datasets.

    Args:
        config_path: Ruta al archivo JSON de configuración.
    """
    cfg = _load_json(config_path)

    # El JSON puede mantenerse mínimo. Si no se define log_fn, se usa print.
    log_fn = _resolve_log_fn(cfg.get("log_fn", "print"))

    input_file_names = _read_input_file_names(cfg)
    output_file_name = _read_output_file_name(cfg)

    summary = merge_tensor_datasets_work(
        input_file_names=input_file_names,
        output_file_name=output_file_name,

        # Defaults operativos: se dejan fuera del JSON para mantenerlo simple.
        overwrite=False,
        keep_names=True,
        verbose=True,
        verbose_storage=True,
        map_location="cpu",
        log_fn=log_fn,
    )

    if log_fn is not None:
        log_fn("\n[OK] Datasets unidos correctamente.")
        log_fn(f"  output_file_name: {summary['output_file_name']}")
        log_fn(f"  output_path: {summary['output_path']}")
        log_fn(f"  total_samples: {summary['total_samples']}")
        log_fn(f"  n_cities: {summary['n_cities']}")
        log_fn(f"  m_items: {summary['m_items']}")
        log_fn(f"  elapsed_s: {summary['elapsed_s']:.3f}")


def main(argv: list[str] | None = None) -> None:
    """Parsea argumentos CLI y ejecuta la unión de datasets.

    Args:
        argv: Argumentos opcionales para testing. Si es ``None``, usa
            ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Une varios datasets TTP tensoriales en un único archivo.",
    )
    parser.add_argument(
        "--config",
        default="configs/datasets/merge_dataset.json",
        help="Ruta al JSON de configuración.",
    )

    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()