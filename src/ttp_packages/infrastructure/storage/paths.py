#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/paths.py

"""Construcción centralizada de rutas del proyecto.

Este módulo define helpers para construir rutas absolutas de instancias,
datasets, checkpoints, historiales, resultados de comparación, estudios Optuna
y plots. La lógica de nombres de archivo vive en ``file_names.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .paths_config import (
    INSTANCES_DIR, 
    TRAIN_DATA_DIR,
    TRAINED_MODEL_DIR, 
    TRAINED_MODEL_PARAMS_DIR,
    TRAINED_MODEL_HISTORY_DIR, 
    TRAINED_MODEL_PLOTS_DIR,
    OPTUNA_DATA_DIR,
    RESULTS_COMPARE_DIR
)
from .file_names import (
    build_history_file_name,
    build_model_checkpoint_name,
    ensure_model_prefix,
    slug,
)

# =============================================================================
# Rutas Base (Instancias, Dataset y compare)
# =============================================================================
def get_inst_file_path(fname: str) -> Path:
    """Construye la ruta absoluta para un archivo de instancia TTP.

    La carpeta se infiere a partir del prefijo del archivo. Por ejemplo,
    ``"a280_n279.ttp"`` se busca dentro de ``"a280-ttp"``.

    Args:
        fname: Nombre del archivo de instancia.

    Returns:
        Ruta absoluta al archivo de instancia.
    """
    prefix = fname.split('_')[0]
    return INSTANCES_DIR / f"{prefix}-ttp" / fname

def build_dataset_path(file_name: str) -> Path:
    """Construye la ruta absoluta para un dataset de entrenamiento.

    Args:
        file_name: Nombre del archivo de dataset.

    Returns:
        Ruta absoluta dentro del directorio de datasets.
    """
    return TRAIN_DATA_DIR / file_name

def build_compare_results_path(file_name: str) -> Path:
    """Construye la ruta absoluta para resultados de comparación.

    Args:
        file_name: Nombre del archivo JSON, PNG u otra salida de comparación.

    Returns:
        Ruta absoluta dentro del directorio de resultados de comparación.
    """
    return RESULTS_COMPARE_DIR / file_name

# =============================================================================
# Rutas de Modelos (Checkpoints, Params, History)
# =============================================================================
def build_model_checkpoint_path(model_id: str, model_params: Optional[Mapping[str, Any]] = None, *, 
                                keys_order: Optional[Sequence[str]] = None) -> Path:
    """Construye la ruta absoluta para un nuevo checkpoint de modelo.

    Args:
        model_id: Identificador base del modelo.
        model_params: Parámetros usados para construir el nombre del checkpoint.
        keys_order: Orden preferido de claves al generar tokens del nombre.

    Returns:
        Ruta absoluta dentro del directorio de modelos entrenados.
    """
    name = build_model_checkpoint_name(model_id, model_params, keys_order=keys_order)
    return TRAINED_MODEL_DIR / name

def resolve_model_checkpoint_path(ckpt_file: str) -> Path:
    """Resuelve la ruta absoluta de un checkpoint existente.

    Args:
        ckpt_file: Nombre exacto del archivo ``.pt`` guardado.

    Returns:
        Ruta absoluta al checkpoint dentro del directorio de modelos entrenados.
    """
    return TRAINED_MODEL_DIR / ckpt_file

def build_params_dir_path(model_id: str) -> Path:
    """Construye la ruta al directorio de parámetros/runs de un modelo.

    Args:
        model_id: Identificador del modelo.

    Returns:
        Ruta absoluta al subdirectorio de parámetros del modelo.
    """
    mid = ensure_model_prefix(model_id)
    return TRAINED_MODEL_PARAMS_DIR / f"{mid}_trained_params"

def build_runs_index_path(model_id: str) -> Path:
    """Construye la ruta al índice maestro de runs de un modelo.

    Args:
        model_id: Identificador del modelo.

    Returns:
        Ruta absoluta al archivo ``runs.json``.
    """
    return build_params_dir_path(model_id) / "runs.json"

def build_run_json_path(model_id: str, run_tag: str) -> Path:
    """Construye la ruta al JSON individual de una run.

    Args:
        model_id: Identificador del modelo.
        run_tag: Etiqueta de la run, por ejemplo ``"run01"``.

    Returns:
        Ruta absoluta al archivo JSON de la run.
    """
    return build_params_dir_path(model_id) / f"{run_tag}.json"

def build_history_dir_path(model_id: str) -> Path:
    """Construye la ruta al directorio de historiales de un modelo.

    Args:
        model_id: Identificador del modelo.

    Returns:
        Ruta absoluta al directorio de historiales.
    """
    mid = ensure_model_prefix(model_id)
    return TRAINED_MODEL_HISTORY_DIR / f"{mid}_history"

def build_history_file_path(model_id: str, run_tag: str) -> Path:
    """Construye la ruta absoluta al archivo de historial de una run.

    Args:
        model_id: Identificador del modelo.
        run_tag: Etiqueta de la run, por ejemplo ``"run01"``.

    Returns:
        Ruta absoluta al archivo JSON de historial.
    """
    return build_history_dir_path(model_id) / build_history_file_name(run_tag)

# =============================================================================
# Rutas de Optuna
# ============================================================================
def build_optuna_db_path(study_name: str) -> Path:
    """Construye la ruta absoluta del archivo SQLite de un estudio Optuna.

    Args:
        study_name: Nombre lógico del estudio.

    Returns:
        Ruta absoluta al archivo ``.db`` del estudio.

    Raises:
        ValueError: Si ``study_name`` está vacío después de normalizarlo.
    """
    sname = slug(study_name)
    if not sname:
        raise ValueError("study_name no puede ser vacío")

    return OPTUNA_DATA_DIR / f"{sname}.db"


# =============================================================================
# Rutas de Plots
# =============================================================================
def build_plot_loss_path(model_id: str, run_tag: str) -> Path:
    """Construye la ruta absoluta para el gráfico de pérdida.

    Args:
        model_id: Identificador del modelo.
        run_tag: Etiqueta de la run, por ejemplo ``"run01"``.

    Returns:
        Ruta absoluta al archivo PNG de pérdida.
    """
    mid = ensure_model_prefix(model_id)
    return TRAINED_MODEL_PLOTS_DIR / f"{mid}_{run_tag}_Loss.png"

def build_plot_acc_path(model_id: str, run_tag: str) -> Path:
    """Construye la ruta absoluta para el gráfico de precisión.

    Args:
        model_id: Identificador del modelo.
        run_tag: Etiqueta de la run, por ejemplo ``"run01"``.

    Returns:
        Ruta absoluta al archivo PNG de precisión.
    """
    mid = ensure_model_prefix(model_id)
    return TRAINED_MODEL_PLOTS_DIR / f"{mid}_{run_tag}_Accuracy.png"