#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/model_out.py

"""Exportación de checkpoints, historiales y metadatos de entrenamiento.

Este módulo persiste artefactos de entrenamiento sin modificar el formato legacy
del proyecto: registros de runs, checkpoints PyTorch, summaries, parámetros de
modelo, parámetros de entrenamiento e historiales por época.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .keys import (
    KEY_CHECKPOINT_FILE,
    KEY_HISTORY,
    KEY_HISTORY_FILE,
    KEY_MODEL_ID,
    KEY_MODEL_PARAMS,
    KEY_RUN_TAG,
    KEY_RUNS,
    KEY_STATE_DICT,
    KEY_SUMMARY,
    KEY_TORCH_VERSION,
    KEY_TRAIN_PARAMS,
)
from .paths import (
    build_history_dir_path,
    build_model_checkpoint_path,
    build_params_dir_path,
    build_run_json_path,
    build_runs_index_path,
)
from .file_names import build_history_file_name, ensure_model_prefix
from .json_io import dump_json, load_json, to_jsonable
from .runs_io import run_id_from_tag, run_tag_from_int
from .torch_io import atomic_torch_save

# =============================================================================
# Funciones Internas de Ayuda (Helpers)
# =============================================================================

def _infer_model_params(model: torch.nn.Module) -> Dict[str, Any]:
    """Extrae parámetros de arquitectura desde un modelo.

    Prioriza el atributo ``config`` si existe. Si no está disponible, intenta
    recuperar manualmente atributos conocidos.

    Args:
        model: Modelo PyTorch.

    Returns:
        Diccionario serializable con parámetros de arquitectura.
    """
    config = getattr(model, "config", None)
    if config is not None:
        return to_jsonable(config)

    # Fallback legacy: modelos antiguos pueden no exponer un objeto config.
    params: Dict[str, Any] = {}
    keys = (
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "dropout",
        "sinkhorn_iter",
        "sink_tau",
        "coupling_iters",
    )

    for key in keys:
        if hasattr(model, key):
            params[key] = getattr(model, key)

    return to_jsonable(params)

def _allocate_run(*, model_id: str) -> tuple[str, str, Path, Path, Path, dict, list]:
    """Reserva una nueva run para un modelo.

    Lee el índice actual, calcula el siguiente ``runXX``, crea el registro base
    y lo agrega a la lista de runs.

    Args:
        model_id: Identificador del modelo.

    Returns:
        Tupla con ``model_id`` normalizado, tag de run, rutas relevantes,
        registro de run y lista completa de runs.
    """
    mid = ensure_model_prefix(model_id)
    root = build_params_dir_path(mid)
    index_path = build_runs_index_path(mid)
    root.mkdir(parents=True, exist_ok=True)

    blob = load_json(index_path) if index_path.exists() else {}
    runs: list = list(blob.get(KEY_RUNS, [])) if isinstance(blob, dict) else []

    max_id = 0
    for record in runs:
        current_id = run_id_from_tag(record.get(KEY_RUN_TAG, ""))
        if current_id >= 0:
            max_id = max(max_id, current_id)

    run_tag = run_tag_from_int(max_id + 1)
    run_path = build_run_json_path(mid, run_tag)

    record = {
        KEY_MODEL_ID: mid,
        KEY_RUN_TAG: run_tag,
        KEY_CHECKPOINT_FILE: None,
        KEY_HISTORY_FILE: None,
        KEY_MODEL_PARAMS: None,
        KEY_TRAIN_PARAMS: None,
        KEY_SUMMARY: None,
    }
    runs.append(record)

    dump_json(run_path, record)
    dump_json(index_path, {KEY_MODEL_ID: mid, KEY_RUNS: runs})

    return mid, run_tag, root, index_path, run_path, record, runs

def _finalize_run(*, mid: str, index_path: Path, run_path: Path, rec: dict, runs: list) -> None:
    """Guarda los metadatos finales de una run.

    Args:
        mid: Identificador normalizado del modelo.
        index_path: Ruta al índice maestro ``runs.json``.
        run_path: Ruta al JSON individual de la run.
        rec: Registro actualizado de la run.
        runs: Lista completa de runs del modelo.
    """
    dump_json(run_path, rec)
    dump_json(index_path, {KEY_MODEL_ID: mid, KEY_RUNS: runs})

def _export_run(
    *,
    model_id: str,
    do_ckpt: bool,
    do_hist: bool,
    model: Optional[torch.nn.Module] = None,
    model_params: Optional[Any] = None,
    train_params: Optional[Any] = None,
    summary: Optional[Mapping[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    checkpoint_file: Optional[str] = None,
    history_file: Optional[str] = None,
    overwrite_model: bool = True,
    overwrite_history: bool = False,
    keys_order: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Exporta artefactos de una run y actualiza sus metadatos.

    Según los flags recibidos, registra una run lógica, guarda checkpoint,
    guarda historial o enlaza artefactos ya existentes.

    Args:
        model_id: Identificador del modelo.
        do_ckpt: Si es True, guarda checkpoint físico.
        do_hist: Si es True, guarda historial físico.
        model: Modelo PyTorch requerido cuando ``do_ckpt=True``.
        model_params: Parámetros de arquitectura. Puede ser mapping o dataclass.
        train_params: Parámetros de entrenamiento.
        summary: Resumen del entrenamiento.
        history: Historial por época.
        checkpoint_file: Archivo de checkpoint existente cuando ``do_ckpt=False``.
        history_file: Archivo de historial existente cuando ``do_hist=False``.
        overwrite_model: Si es True, permite sobrescribir checkpoint existente.
        overwrite_history: Si es True, permite sobrescribir historial existente.
        keys_order: Orden preferido de claves para nombrar el checkpoint.

    Returns:
        Diccionario con rutas generadas.

    Raises:
        ValueError: Si se solicita checkpoint sin modelo.
        FileExistsError: Si un artefacto ya existe y no se permite sobrescritura.
    """
    mid, run_tag, root, index_path, run_path, rec, runs = _allocate_run(model_id=model_id)

    ckpt_path: Optional[Path] = None
    hist_path: Optional[Path] = None

    if do_ckpt:
        if model is None: 
            raise ValueError("model es requerido para exportar checkpoint")
        
        # Normaliza dataclasses, enums y tensores antes de usar los parámetros en el nombre.
        if model_params is not None:
            mp = to_jsonable(model_params)
        else:
            mp = _infer_model_params(model)
        
        if not isinstance(mp, dict):
            raise ValueError(
                "model_params debe convertirse a un diccionario para nombrar el checkpoint."
            )
        
        ckpt_path = build_model_checkpoint_path(mid, mp, keys_order=keys_order)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        
        if ckpt_path.exists() and not overwrite_model:
            raise FileExistsError(f"Ya existe checkpoint: {ckpt_path} (usa overwrite_model=True)")
        
        tv = getattr(torch, "__version__", None)
        atomic_torch_save(
            {
                KEY_MODEL_ID: mid,
                KEY_RUN_TAG: run_tag,
                KEY_STATE_DICT: model.state_dict(),
                KEY_MODEL_PARAMS: to_jsonable(mp),
                KEY_TRAIN_PARAMS: to_jsonable(train_params) if train_params is not None else None,
                KEY_SUMMARY: to_jsonable(summary) if summary is not None else None,
                KEY_TORCH_VERSION: None if tv is None else str(tv),
            },
            ckpt_path,
        )
        
        rec[KEY_CHECKPOINT_FILE] = ckpt_path.name
        rec[KEY_MODEL_PARAMS] = to_jsonable(mp)
        rec[KEY_TRAIN_PARAMS] = to_jsonable(train_params) if train_params is not None else None
        rec[KEY_SUMMARY] = to_jsonable(summary) if summary is not None else None
    else:
        rec[KEY_CHECKPOINT_FILE] = checkpoint_file
        rec[KEY_MODEL_PARAMS] = to_jsonable(model_params) if model_params is not None else None
        rec[KEY_TRAIN_PARAMS] = to_jsonable(train_params) if train_params is not None else None
        rec[KEY_SUMMARY] = to_jsonable(summary) if summary is not None else None

    if do_hist:
        hroot = build_history_dir_path(mid)
        hroot.mkdir(parents=True, exist_ok=True)
        
        hist_path = hroot / build_history_file_name(run_tag)
        if hist_path.exists() and not overwrite_history:
            raise FileExistsError(f"Ya existe history: {hist_path} (usa overwrite_history=True)")
            
        dump_json(
            hist_path, 
            {
                KEY_MODEL_ID: mid, 
                KEY_RUN_TAG: run_tag, 
                KEY_HISTORY: history or []
            },
        )
        rec[KEY_HISTORY_FILE] = hist_path.name
    else:
        rec[KEY_HISTORY_FILE] = history_file

    _finalize_run(mid=mid, index_path=index_path, run_path=run_path, rec=rec, runs=runs)

    out: Dict[str, Path] = {"params_dir": root, "params_index": index_path, "params_run": run_path}
    if ckpt_path is not None: 
        out["checkpoint"] = ckpt_path
    if hist_path is not None: 
        out["history"] = hist_path
    return out

# =============================================================================
# API Pública
# =============================================================================

def export_params_run(
    *,
    model_id: str,
    model_params: Optional[Any] = None,
    train_params: Optional[Any] = None,
    summary: Optional[Mapping[str, Any]] = None,
    checkpoint_file: Optional[str] = None,
    history_file: Optional[str] = None,
) -> Dict[str, Path]:
    """Crea un registro lógico de run sin guardar pesos ni historial.

    Esta función registra metadata de una run y puede enlazar artefactos ya
    existentes mediante ``checkpoint_file`` y ``history_file``.

    Args:
        model_id: Identificador del modelo.
        model_params: Parámetros de arquitectura o estructura serializable.
        train_params: Parámetros de entrenamiento o estructura serializable.
        summary: Resumen opcional del entrenamiento.
        checkpoint_file: Nombre de un checkpoint existente.
        history_file: Nombre de un historial existente.

    Returns:
        Diccionario con rutas de metadata generadas.
    """
    return _export_run(
        model_id=model_id,
        model_params=model_params,
        train_params=train_params,
        summary=summary,
        checkpoint_file=checkpoint_file,
        history_file=history_file,
        do_ckpt=False,
        do_hist=False,
    )

def export_model_checkpoint(
    *,
    model_id: str,
    model: torch.nn.Module,
    model_params: Optional[Any] = None,
    train_params: Optional[Any] = None,
    summary: Optional[Mapping[str, Any]] = None,
    overwrite_model: bool = True,
    keys_order: Optional[Sequence[str]] = None,
) -> Path:
    """Exporta un checkpoint PyTorch de un modelo entrenado.

    Crea una nueva run, guarda el ``state_dict`` junto con parámetros y metadata,
    y actualiza los índices JSON del modelo.

    Args:
        model_id: Identificador del modelo.
        model: Modelo PyTorch a guardar.
        model_params: Parámetros de arquitectura. Si es ``None``, se infieren
            desde el modelo.
        train_params: Parámetros de entrenamiento.
        summary: Resumen opcional del entrenamiento.
        overwrite_model: Si es True, permite sobrescribir checkpoint existente.
        keys_order: Orden preferido para nombrar el archivo del checkpoint.

    Returns:
        Ruta del checkpoint guardado.
    """
    out = _export_run(
        model_id=model_id,
        model=model,
        model_params=model_params,
        train_params=train_params,
        summary=summary,
        do_ckpt=True,
        do_hist=False,
        overwrite_model=overwrite_model,
        keys_order=keys_order,
    )
    return out["checkpoint"]

def export_history_run(
    *,
    model_id: str,
    history: Optional[List[Dict[str, Any]]] = None,
    overwrite_history: bool = False,
) -> Path:
    """Exporta un historial métrico como JSON.

    Crea una nueva run lógica y guarda las métricas por época en un archivo
    ``.history.json``.

    Args:
        model_id: Identificador del modelo.
        history: Lista de métricas por época.
        overwrite_history: Si es True, permite sobrescribir un historial existente.

    Returns:
        Ruta absoluta al historial JSON generado.

    Raises:
        FileExistsError: Si el historial ya existe y ``overwrite_history`` es False.
    """
    out = _export_run(
        model_id=model_id, history=history, do_ckpt=False, do_hist=True, 
        overwrite_history=overwrite_history,
    )
    return out["history"]

def export_training_artifacts(
    *,
    model_id: str,
    model: torch.nn.Module,
    model_params: Optional[Any] = None,
    train_params: Optional[Any] = None,
    summary: Optional[Mapping[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    overwrite_model: bool = True,
    overwrite_history: bool = False,
    keys_order: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Exporta todos los artefactos de una sesión de entrenamiento.

    Guarda el checkpoint PyTorch, el historial de entrenamiento y los metadatos
    de la run usando el mismo ``run_tag`` para mantener trazabilidad.

    Args:
        model_id: Identificador del modelo.
        model: Modelo PyTorch a guardar.
        model_params: Parámetros de arquitectura. Si es ``None``, se infieren
            desde el modelo.
        train_params: Parámetros de entrenamiento.
        summary: Resumen opcional del entrenamiento.
        history: Métricas del entrenamiento por época.
        overwrite_model: Si es True, permite sobrescribir checkpoint existente.
        overwrite_history: Si es True, permite sobrescribir historial existente.
        keys_order: Orden preferido para nombrar el checkpoint.

    Returns:
        Diccionario con rutas generadas, como ``checkpoint``, ``history``,
        ``params_dir``, ``params_index`` y ``params_run``.

    Raises:
        ValueError: Si falta el modelo o los parámetros no pueden convertirse a
            diccionario.
        FileExistsError: Si un artefacto ya existe y no se permite sobrescritura.
    """

    return _export_run(
        model_id=model_id,
        model=model,
        model_params=model_params,
        train_params=train_params,
        summary=summary,
        history=history,
        do_ckpt=True,
        do_hist=True,
        overwrite_model=overwrite_model,
        overwrite_history=overwrite_history,
        keys_order=keys_order,
    )