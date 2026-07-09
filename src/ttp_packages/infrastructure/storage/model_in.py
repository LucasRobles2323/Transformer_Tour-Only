#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/model_in.py

"""Carga de modelos entrenados y artefactos asociados.

Este módulo recupera runs registradas, summaries, parámetros de entrenamiento,
historiales y checkpoints PyTorch. También reconstruye modelos TTP a partir de
los metadatos guardados.

La estructura de los checkpoints y JSON existentes se mantiene sin cambios para
conservar compatibilidad con modelos ya entrenados.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch

from src.ttp_packages.infrastructure.runtime import get_default_map_location

from .keys import (
    KEY_CHECKPOINT_FILE,
    KEY_HISTORY,
    KEY_HISTORY_FILE,
    KEY_MODEL_ID,
    KEY_MODEL_PARAMS,
    KEY_RUN_TAG,
    KEY_STATE_DICT,
    KEY_SUMMARY,
    KEY_TORCH_VERSION,
    KEY_TRAIN_PARAMS,
)
from .paths import (
    build_history_dir_path,
    resolve_model_checkpoint_path,
)
from .json_io import load_json
from .runs_io import read_runs_index, select_run_record
from .torch_io import load_torch_dict

if TYPE_CHECKING:
    from src.ttp_packages.modeling.ttp_model import TTPModel


def import_summary(
    *,
    model_id: str,
    run_id: Optional[Union[int, str]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Optional[Dict[str, Any]]:
    """Importa el summary de una run de entrenamiento.

    Primero busca el summary en el registro de la run. Si no está ahí, intenta
    recuperarlo desde el checkpoint asociado.

    Args:
        model_id: Identificador del modelo.
        run_id: Identificador opcional de la run. Si es ``None``, usa la más
            reciente.
        map_location: Dispositivo destino para cargar checkpoint si hiciera falta.

    Returns:
        Diccionario de summary, o ``None`` si no existe.
    """
    _model_id_norm, _root, runs = read_runs_index(model_id=model_id)
    record = select_run_record(runs, run_id)

    summary = record.get(KEY_SUMMARY, None)
    if isinstance(summary, dict):
        return summary

    checkpoint_file = record.get(KEY_CHECKPOINT_FILE, None)
    if not checkpoint_file:
        return None

    checkpoint_path = resolve_model_checkpoint_path(str(checkpoint_file)).resolve()
    if not checkpoint_path.exists():
        return None

    if map_location is None:
        map_location = get_default_map_location()

    try:
        checkpoint = load_torch_dict(checkpoint_path, map_location=map_location)
    except ValueError:
        return None

    summary_from_ckpt = checkpoint.get(KEY_SUMMARY, None)
    return summary_from_ckpt if isinstance(summary_from_ckpt, dict) else None


def import_train_params(
    *,
    model_id: str,
    run_id: Optional[Union[int, str]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Optional[Dict[str, Any]]:
    """Importa los parámetros de entrenamiento de una run.

    Primero busca los parámetros en el registro de la run. Si no están ahí,
    intenta recuperarlos desde el checkpoint asociado.

    Args:
        model_id: Identificador del modelo.
        run_id: Identificador opcional de la run. Si es ``None``, usa la más
            reciente.
        map_location: Dispositivo destino para cargar checkpoint si hiciera falta.

    Returns:
        Diccionario de parámetros de entrenamiento, o ``None`` si no existe.
    """
    _model_id_norm, _root, runs = read_runs_index(model_id=model_id)
    record = select_run_record(runs, run_id)

    train_params = record.get(KEY_TRAIN_PARAMS, None)
    if isinstance(train_params, dict):
        return train_params

    checkpoint_file = record.get(KEY_CHECKPOINT_FILE, None)
    if not checkpoint_file:
        return None

    checkpoint_path = resolve_model_checkpoint_path(str(checkpoint_file)).resolve()
    if not checkpoint_path.exists():
        return None

    if map_location is None:
        map_location = get_default_map_location()

    try:
        checkpoint = load_torch_dict(checkpoint_path, map_location=map_location)
    except ValueError:
        return None

    train_params_from_ckpt = checkpoint.get(KEY_TRAIN_PARAMS, None)
    return train_params_from_ckpt if isinstance(train_params_from_ckpt, dict) else None


def import_history(
    *,
    model_id: str,
    run_id: Optional[Union[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Importa el historial métrico de una run de entrenamiento.

    Args:
        model_id: Identificador del modelo.
        run_id: Identificador opcional de la run. Si es ``None``, usa la más
            reciente.

    Returns:
        Lista de registros por época. Retorna lista vacía si no hay historial.
    """
    model_id_norm, _root, runs = read_runs_index(model_id=model_id)
    record = select_run_record(runs, run_id)

    run_tag = str(record.get(KEY_RUN_TAG, "")).strip()
    history_file = record.get(KEY_HISTORY_FILE, None)

    if not history_file and run_tag:
        history_file = f"{run_tag}.history.json"

    if not history_file:
        return []

    history_root = build_history_dir_path(model_id_norm)
    history_path = (history_root / str(history_file)).resolve()

    if not history_path.exists():
        return []

    blob = load_json(history_path)
    history = blob.get(KEY_HISTORY, []) if isinstance(blob, dict) else []

    return list(history) if isinstance(history, list) else []


def import_model(
    *,
    model_id: str,
    run_id: Optional[Union[int, str]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    eval_mode: bool = True,
) -> Tuple["TTPModel", Dict[str, Any]]:
    """Reconstruye y carga un modelo TTP desde una run registrada.

    Args:
        model_id: Identificador del modelo.
        run_id: Identificador opcional de la run. Si es ``None``, usa la más
            reciente.
        map_location: Dispositivo usado para cargar tensores desde disco.
        device: Dispositivo final donde se moverá el modelo cargado.
        strict: Si es True, exige coincidencia estricta del ``state_dict``.
        eval_mode: Si es True, deja el modelo en modo evaluación.

    Returns:
        Tupla ``(model, meta)`` con el modelo reconstruido y metadatos de carga.

    Raises:
        FileNotFoundError: Si no existe checkpoint registrado o físico.
        ValueError: Si faltan ``model_params`` o ``state_dict``.
    """
    # Import local: solo se requiere modelado cuando realmente se reconstruye.
    from src.ttp_packages.modeling.config import TTPArchitectureParams
    from src.ttp_packages.modeling.ttp_model import TTPModel

    if map_location is None:
        map_location = get_default_map_location()

    model_id_norm, _root, runs = read_runs_index(model_id=model_id)
    record = select_run_record(runs, run_id)

    checkpoint_file = record.get(KEY_CHECKPOINT_FILE, None)
    if not checkpoint_file:
        raise FileNotFoundError(
            f"No hay checkpoint_file registrado para '{model_id_norm}'."
        )

    checkpoint_path = resolve_model_checkpoint_path(str(checkpoint_file)).resolve()
    checkpoint = load_torch_dict(checkpoint_path, map_location=map_location)

    model_params_dict = checkpoint.get(KEY_MODEL_PARAMS, None)
    if not isinstance(model_params_dict, dict):
        model_params_dict = record.get(KEY_MODEL_PARAMS, None)

    if not isinstance(model_params_dict, dict):
        raise ValueError("No se encontraron model_params para reconstruir el modelo.")

    model_params_obj = TTPArchitectureParams(**model_params_dict)
    model = TTPModel(params=model_params_obj)

    state = checkpoint.get(KEY_STATE_DICT, None)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint sin state_dict: {checkpoint_path}")

    model.load_state_dict(state, strict=strict)

    if device is not None:
        model = model.to(device)

    if eval_mode:
        model.eval()

    summary = record.get(KEY_SUMMARY, None)
    if not isinstance(summary, dict):
        summary = checkpoint.get(KEY_SUMMARY, None)

    if not isinstance(summary, dict):
        summary = None

    meta: Dict[str, Any] = {
        KEY_MODEL_ID: model_id_norm,
        KEY_RUN_TAG: record.get(KEY_RUN_TAG, None),
        "checkpoint_path": str(checkpoint_path),
        KEY_MODEL_PARAMS: model_params_dict,
        KEY_TRAIN_PARAMS: checkpoint.get(KEY_TRAIN_PARAMS, None),
        KEY_SUMMARY: summary,
        KEY_TORCH_VERSION: checkpoint.get(KEY_TORCH_VERSION, None),
    }

    train_params = checkpoint.get(KEY_TRAIN_PARAMS, None)
    if not isinstance(train_params, dict):
        train_params = record.get(KEY_TRAIN_PARAMS, None)

    if train_params is None:
        train_params = {}

    setattr(model, "train_params", train_params)
    setattr(model, "training_summary", summary)
    setattr(model, "model_meta", meta)

    return model, meta


def import_training_artifacts(
    *,
    model_id: str,
    run_id: Optional[Union[int, str]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    eval_mode: bool = True,
) -> Dict[str, Any]:
    """Importa modelo, parámetros, summary e historial de una run.

    Args:
        model_id: Identificador del modelo.
        run_id: Identificador opcional de la run. Si es ``None``, usa la más
            reciente.
        map_location: Dispositivo usado al cargar tensores desde disco.
        device: Dispositivo final del modelo cargado.
        strict: Si es True, carga estricta del ``state_dict``.
        eval_mode: Si es True, deja el modelo en modo evaluación.

    Returns:
        Diccionario con ``model``, ``history``, ``train_params``, ``summary`` y
        ``meta``.
    """
    if map_location is None:
        map_location = get_default_map_location()

    model, meta = import_model(
        model_id=model_id,
        run_id=run_id,
        map_location=map_location,
        device=device,
        strict=strict,
        eval_mode=eval_mode,
    )

    train_params = import_train_params(
        model_id=model_id,
        run_id=run_id,
        map_location=map_location,
    )
    summary = import_summary(
        model_id=model_id,
        run_id=run_id,
        map_location=map_location,
    )
    history = import_history(
        model_id=model_id,
        run_id=run_id,
    )

    if train_params is None:
        train_params = {}

    setattr(model, "train_params", train_params)
    setattr(model, "training_summary", summary)
    setattr(model, "model_meta", meta)

    return {
        "model": model,
        KEY_HISTORY: history,
        KEY_TRAIN_PARAMS: train_params,
        KEY_SUMMARY: summary,
        "meta": meta,
    }


def import_model_from_checkpoint_file(
    *,
    checkpoint_file: str,
    map_location: Optional[Union[str, torch.device]] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    eval_mode: bool = True,
) -> Tuple["TTPModel", Dict[str, Any]]:
    """Reconstruye y carga un modelo TTP directamente desde un checkpoint.

    Args:
        checkpoint_file: Nombre del archivo ``.pt`` resoluble por storage.
        map_location: Dispositivo usado para cargar tensores desde disco.
        device: Dispositivo final donde se moverá el modelo cargado.
        strict: Si es True, exige coincidencia estricta del ``state_dict``.
        eval_mode: Si es True, deja el modelo en modo evaluación.

    Returns:
        Tupla ``(model, meta)`` con el modelo reconstruido y metadatos.

    Raises:
        FileNotFoundError: Si el checkpoint no existe.
        ValueError: Si faltan ``model_params`` o ``state_dict``.
    """
    # Import local: evita cargar modeling al importar solo helpers de metadata.
    from src.ttp_packages.modeling.config import TTPArchitectureParams
    from src.ttp_packages.modeling.ttp_model import TTPModel

    if map_location is None:
        map_location = get_default_map_location()

    checkpoint_path = resolve_model_checkpoint_path(str(checkpoint_file)).resolve()
    checkpoint = load_torch_dict(checkpoint_path, map_location=map_location)

    model_params_dict = checkpoint.get(KEY_MODEL_PARAMS, None)
    if not isinstance(model_params_dict, dict):
        raise ValueError(
            f"El checkpoint no contiene '{KEY_MODEL_PARAMS}' para reconstruir el modelo."
        )

    model_params_obj = TTPArchitectureParams(**model_params_dict)
    model = TTPModel(params=model_params_obj)

    state = checkpoint.get(KEY_STATE_DICT, None)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint sin state_dict: {checkpoint_path}")

    model.load_state_dict(state, strict=strict)

    if device is not None:
        model = model.to(device)

    if eval_mode:
        model.eval()

    meta: Dict[str, Any] = {
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_path": str(checkpoint_path),
        KEY_MODEL_PARAMS: model_params_dict,
        KEY_TRAIN_PARAMS: checkpoint.get(KEY_TRAIN_PARAMS, None),
        KEY_SUMMARY: checkpoint.get(KEY_SUMMARY, None),
        KEY_TORCH_VERSION: checkpoint.get(KEY_TORCH_VERSION, None),
    }

    train_params = checkpoint.get(KEY_TRAIN_PARAMS, None)
    if train_params is None:
        train_params = {}

    setattr(model, "train_params", train_params)
    setattr(model, "training_summary", checkpoint.get(KEY_SUMMARY, None))
    setattr(model, "model_meta", meta)

    return model, meta