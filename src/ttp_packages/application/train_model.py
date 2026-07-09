#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/train_model.py

"""Workflows de entrenamiento del modelo tour-only.

Este módulo encapsula carga de dataset, construcción del modelo, entrenamiento,
exportación de artefactos y exportación opcional de plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from . import config as cfg
from .build_model import instantiate_ttp_model

from src.ttp_packages.infrastructure.storage.model_out import export_training_artifacts
from src.ttp_packages.ml_data.torch.dataset import TTPTensorDataset
from src.ttp_packages.training.engine import train_tour_only


def train_tour_only_work(
    model: torch.nn.Module,
    dataset: Any,
    train_params: cfg.TrainingParams = cfg.DEFAULT_TRAIN_PARAMS,
    log_fn: Optional[Callable[[str], None]] = print,
) -> Tuple[torch.nn.Module, list[Dict[str, Any]], Dict[str, Any]]:
    """Entrena un modelo tour-only con parámetros de entrenamiento.

    Args:
        model: Modelo neuronal a entrenar.
        dataset: Dataset tensorial TTP.
        train_params: Parámetros de entrenamiento.
        log_fn: Función opcional de logging.

    Returns:
        Tupla ``(trained_model, history, summary)``.
    """
    trained_model, history, summary = train_tour_only(
        model=model,
        dataset=dataset,
        params=train_params,
        log_fn=log_fn,
    )

    return trained_model, history, summary


def export_training_plots_work(
    *,
    model_id: str,
    run_id: Optional[Union[int, str]] = None,
    dpi: int = 160,
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = print,
) -> Dict[str, Path]:
    """Exporta plots de entrenamiento para un modelo.

    Args:
        model_id: Identificador del modelo.
        run_id: Corrida específica a graficar. Si es ``None``, usa la última.
        dpi: Resolución de los plots exportados.
        verbose: Si es ``True``, registra las rutas generadas.
        log_fn: Función opcional de logging.

    Returns:
        Diccionario con rutas de plots exportados.
    """
    # Import local: evita cargar dependencias de plotting si no se exportan plots.
    from src.ttp_packages.infrastructure.storage.plot_io import export_training_plots

    out = export_training_plots(model_id=model_id, run_id=run_id, dpi=dpi)

    if verbose and log_fn is not None:
        log_fn(f"[PLOTS] Perdida   -> {out['loss']}")
        log_fn(f"[PLOTS] Presicion -> {out['accuracy']}")

    return out


def fit_tour_model_work(
    *,
    dataset_file: str,
    model_id: str,
    model_params: cfg.TTPArchitectureParams = cfg.DEFAULT_MODEL_PARAMS,
    train_params: cfg.TrainingParams = cfg.DEFAULT_TRAIN_PARAMS,
    export_plots: bool = True,
    dpi: int = cfg.DEFAULT_PLOT_DPI,
    overwrite_model: bool = True,
    overwrite_history: bool = False,
    log_fn: Optional[Callable[[str], None]] = print,
) -> Dict[str, Any]:
    """Ejecuta el workflow completo de entrenamiento tour-only.

    Args:
        dataset_file: Archivo del dataset tensorial.
        model_id: Identificador bajo el que se exportan artefactos.
        model_params: Parámetros de arquitectura del modelo.
        train_params: Parámetros de entrenamiento.
        export_plots: Si es ``True``, exporta plots al final.
        dpi: Resolución de los plots exportados.
        overwrite_model: Si es ``True``, permite sobrescribir el modelo.
        overwrite_history: Si es ``True``, permite sobrescribir historial.
        log_fn: Función opcional de logging.

    Returns:
        Diccionario con ``summary``, ``run_tag``, rutas de artefactos y rutas de
        plots.
    """
    if log_fn is None:
        log_fn = print

    # Carga el dataset en CPU para que el trainer controle el device después.
    dataset = TTPTensorDataset.from_file(
        dataset_file,
        verbose=False,
        map_location="cpu",
    )

    model = instantiate_ttp_model(
        params=model_params,
        device=train_params.device,
        eval_mode=False,
    )

    trained_model, history, summary = train_tour_only_work(
        model=model,
        dataset=dataset,
        train_params=train_params,
        log_fn=log_fn,
    )

    # Guarda el nombre del dataset en el summary, 
    # como primer elemento
    summary = {
        "dataset_file": str(dataset_file),
        **dict(summary),
    }

    export_out = export_training_artifacts(
        model_id=model_id,
        model=trained_model,
        model_params=model_params,
        train_params=train_params,
        summary=summary,
        history=history,
        overwrite_model=overwrite_model,
        overwrite_history=overwrite_history,
    )

    # El stem del archivo de parámetros identifica la corrida exportada.
    run_tag = export_out["params_run"].stem

    plots_out = None
    if export_plots:
        plots_out = export_training_plots_work(
            model_id=model_id,
            run_id=run_tag,
            dpi=dpi,
            verbose=True,
            log_fn=log_fn,
        )

    log_fn("[OK] Entrenamiento finalizado.")
    log_fn(f"  model_id: {model_id}")
    log_fn(f"  best_epoch: {summary.get('best_epoch')}")
    log_fn(f"  best_val_loss: {summary.get('best_val_loss')}")
    log_fn(f"  best_val_acc_at_best_loss: {summary.get('best_val_acc_at_best_loss')}")

    return {
        "summary": summary,
        "run_tag": run_tag,
        "export_paths": export_out,
        "plot_paths": plots_out,
    }