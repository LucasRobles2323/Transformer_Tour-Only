#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/runs_io.py

"""Helpers para leer y seleccionar runs de modelos entrenados."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .json_io import load_json
from .keys import KEY_RUNS, KEY_RUN_TAG, RUN_ID_WIDTH
from .file_names import ensure_model_prefix
from .paths import (
    build_params_dir_path,
    build_runs_index_path,
)


_RUN_TAG_RE = re.compile(r"^run(\d+)$", re.I)


def coerce_run_id(run_id: Optional[Union[int, str]]) -> Optional[int]:
    """Convierte un identificador de run a entero.

    Args:
        run_id: Identificador como entero, string numérico o etiqueta ``runXX``.

    Returns:
        Número de run, o ``None`` si no se pudo interpretar.
    """
    if run_id is None:
        return None

    if isinstance(run_id, int):
        return int(run_id)

    run_id_str = str(run_id).strip()
    if not run_id_str:
        return None

    match = _RUN_TAG_RE.match(run_id_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    try:
        return int(run_id_str)
    except ValueError:
        return None


def run_tag_from_int(run_id: int) -> str:
    """Formatea un entero como etiqueta de run.

    Args:
        run_id: Número de run.

    Returns:
        Etiqueta con formato ``runXX``.
    """
    return f"run{int(run_id):0{RUN_ID_WIDTH}d}"


def run_id_from_tag(run_tag: Any) -> int:
    """Extrae el número entero desde una etiqueta ``runXX``.

    Args:
        run_tag: Etiqueta de run.

    Returns:
        Número de run, o ``-1`` si la etiqueta no es válida.
    """
    match = _RUN_TAG_RE.match(str(run_tag).strip())
    if not match:
        return -1

    try:
        return int(match.group(1))
    except ValueError:
        return -1


def read_runs_index(*, model_id: str) -> Tuple[str, Path, List[dict]]:
    """Lee el índice maestro de runs de un modelo.

    Args:
        model_id: Identificador del modelo.

    Returns:
        Tupla ``(model_id_normalizado, params_dir, runs)``.

    Raises:
        FileNotFoundError: Si no existe el índice o no contiene runs.
    """
    model_id_norm = ensure_model_prefix(model_id)
    params_root = build_params_dir_path(model_id_norm)
    index_path = build_runs_index_path(model_id_norm)

    if not index_path.exists():
        raise FileNotFoundError(
            f"No existe índice de runs para '{model_id_norm}'. "
            f"Se esperaba: {index_path}."
        )

    blob = load_json(index_path)
    runs = list(blob.get(KEY_RUNS, [])) if isinstance(blob, dict) else []

    if not runs:
        raise FileNotFoundError(f"El índice {index_path} no contiene runs.")

    return model_id_norm, params_root, runs


def select_run_record(
    runs: List[dict],
    run_id: Optional[Union[int, str]],
) -> dict:
    """Selecciona una run específica o la más reciente disponible.

    Conserva el comportamiento legacy: si ``run_id`` es ``None`` o no se puede
    resolver de forma válida, retorna la última run identificable.

    Args:
        runs: Lista de registros de runs.
        run_id: Identificador de run, por ejemplo ``1`` o ``"run01"``.

    Returns:
        Registro de la run seleccionada.
    """
    requested_id = coerce_run_id(run_id)
    by_id: Dict[int, dict] = {}

    for record in runs:
        current_id = run_id_from_tag(record.get(KEY_RUN_TAG, ""))
        if current_id >= 0:
            by_id[current_id] = record

    if requested_id is not None and requested_id in by_id:
        return by_id[requested_id]

    if not by_id:
        return runs[-1]

    # Compatibilidad legacy: si no se especifica una run válida, usa la última.
    return by_id[max(by_id.keys())]