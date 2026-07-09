#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/merge_dataset.py

"""Workflow de unión de datasets tensoriales TTP.

Este módulo expone una función de aplicación lista para ser llamada desde
scripts o desde ``main``. La lógica pesada se delega a ``infrastructure`` y la
unión tensorial pura a ``ml_data``.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch

from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.infrastructure.storage.dataset_io import merge_dataset_files
from src.ttp_packages.ml_data.representation.payload import validate_payload


logger = setup_logger(__name__)


def merge_tensor_datasets_work(
    input_file_names: Sequence[str],
    output_file_name: str,
    *,
    overwrite: bool = False,
    keep_names: bool = True,
    verbose: bool = True,
    verbose_storage: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Ejecuta el workflow de unión de datasets tensoriales.

    Args:
        input_file_names: Datasets existentes a unir. No hay límite artificial
            de cantidad; el límite práctico es memoria RAM y tamaño en disco.
        output_file_name: Nombre del nuevo dataset unido.
        overwrite: Si es ``True``, permite sobrescribir el output.
        keep_names: Si es ``True``, conserva o genera ``payload["names"]``.
        verbose: Si es ``True``, registra resumen del workflow.
        verbose_storage: Si es ``True``, registra carga y guardado de datasets.
        map_location: Dispositivo usado al cargar datasets. Si es ``None``,
            carga en CPU.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Diccionario con resumen operativo del merge.

    Raises:
        FileExistsError: Si el output existe y ``overwrite`` es ``False``.
        FileNotFoundError: Si algún input no existe.
        ValueError: Si los datasets son incompatibles.
    """
    if log_fn is None:
        log_fn = logger.info

    if map_location is None:
        map_location = "cpu"

    started_at = perf_counter()

    if verbose:
        log_fn("\n[MERGE_DATASETS_WORK] Inicio")
        log_fn(f"[MERGE_DATASETS_WORK] Inputs: {len(input_file_names)}")
        log_fn(f"[MERGE_DATASETS_WORK] Output: {output_file_name}")

    merged_payload, output_path, input_num_samples = merge_dataset_files(
        input_file_names=input_file_names,
        output_file_name=output_file_name,
        overwrite=overwrite,
        keep_names=keep_names,
        verbose=verbose_storage,
        map_location=map_location,
        log_fn=log_fn,
    )

    total_samples, n_cities, m_items = validate_payload(merged_payload)
    elapsed_s = perf_counter() - started_at

    summary: Dict[str, Any] = {
        "output_file_name": str(output_file_name),
        "output_path": str(output_path),
        "input_file_names": [str(file_name) for file_name in input_file_names],
        "input_num_samples": dict(input_num_samples),
        "total_samples": int(total_samples),
        "n_cities": int(n_cities),
        "m_items": int(m_items),
        "elapsed_s": float(elapsed_s),
        "overwrite": bool(overwrite),
        "keep_names": bool(keep_names),
    }

    if verbose:
        log_fn("\n[MERGE_DATASETS_WORK] Resumen final:")
        log_fn(f"  output_path={summary['output_path']}")
        log_fn(f"  total_samples={summary['total_samples']}")
        log_fn(f"  N={summary['n_cities']} M={summary['m_items']}")
        log_fn(f"  elapsed_s={summary['elapsed_s']:.3f}")
        log_fn("[MERGE_DATASETS_WORK] Fin\n")

    return summary