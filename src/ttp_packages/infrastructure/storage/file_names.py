#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/file_names.py

"""Construcción y normalización de nombres de archivo para storage.

Este módulo contiene helpers para sanitizar tokens, asegurar extensiones y
construir nombres de archivos usados por checkpoints, historiales y salidas.
"""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Optional, Sequence

from .keys import RUN_ID_WIDTH


_MODEL_PREFIX_RE = re.compile(r"^model", re.I)
_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_MULTI_US_RE = re.compile(r"_+")


def ensure_suffix(file_name: str, suffix: str) -> str:
    """Asegura que un nombre de archivo termine con una extensión dada.

    Args:
        file_name: Nombre base o completo del archivo.
        suffix: Extensión esperada, por ejemplo ``".json"`` o ``"png"``.

    Returns:
        Nombre de archivo con la extensión esperada.

    Raises:
        ValueError: Si ``file_name`` o ``suffix`` están vacíos.
    """
    clean_name = str(file_name).strip()
    clean_suffix = str(suffix).strip()

    if not clean_name:
        raise ValueError("file_name no puede estar vacío.")

    if not clean_suffix:
        raise ValueError("suffix no puede estar vacío.")

    if not clean_suffix.startswith("."):
        clean_suffix = f".{clean_suffix}"

    return clean_name if clean_name.endswith(clean_suffix) else f"{clean_name}{clean_suffix}"


def ensure_json_suffix(file_name: str) -> str:
    """Asegura que un nombre de archivo termine en ``.json``."""
    return ensure_suffix(file_name, ".json")


def ensure_png_suffix(file_name: str) -> str:
    """Asegura que un nombre de archivo termine en ``.png``."""
    return ensure_suffix(file_name, ".png")


def ensure_pt_suffix(file_name: str) -> str:
    """Asegura que un nombre de archivo termine en ``.pt``."""
    return ensure_suffix(file_name, ".pt")


def ensure_model_prefix(model_id: str) -> str:
    """Asegura que un identificador de modelo comience con ``model``.

    Args:
        model_id: Identificador original del modelo, por ejemplo ``"01"`` o
            ``"model01"``.

    Returns:
        Identificador con prefijo ``model``.

    Raises:
        ValueError: Si ``model_id`` está vacío.
    """
    model_id = str(model_id).strip()

    if not model_id:
        raise ValueError("model_id no puede estar vacío.")

    return model_id if _MODEL_PREFIX_RE.match(model_id) else f"model{model_id}"


def slug(value: Any) -> str:
    """Convierte un valor en un token seguro para nombres de archivo.

    Args:
        value: Valor a convertir en texto seguro.

    Returns:
        Token sanitizado.
    """
    text = str(value).strip().replace(" ", "_")
    text = _SAFE_CHARS_RE.sub("_", text)
    return _MULTI_US_RE.sub("_", text).strip("_")


def float_token(value: float) -> str:
    """Convierte un float a token seguro para nombres de archivo."""
    return f"{float(value):g}".replace(".", "p")


def tokens_from_params(
    params: Mapping[str, Any],
    *,
    keys_order: Optional[Sequence[str]] = None,
) -> List[str]:
    """Genera tokens seguros a partir de un diccionario de parámetros.

    Args:
        params: Parámetros usados para construir el nombre del archivo.
        keys_order: Orden preferido de claves.

    Returns:
        Lista de tokens sanitizados.
    """
    if not params:
        return []

    used = set()
    keys: List[str] = []

    if keys_order:
        for key in keys_order:
            if key in params:
                keys.append(key)
                used.add(key)

    # Las claves no priorizadas se ordenan para que el nombre sea determinista.
    keys.extend(key for key in sorted(params.keys()) if key not in used)

    tokens: List[str] = []

    for key in keys:
        value = params.get(key)

        if value is None:
            continue

        if isinstance(value, bool):
            value_str = str(int(value))
        elif isinstance(value, int):
            value_str = str(value)
        elif isinstance(value, float):
            value_str = float_token(value)
        else:
            value_str = slug(value)

        token = slug(f"{key}{value_str}")
        if token:
            tokens.append(token)

    return tokens


def build_model_checkpoint_name(
    model_id: str,
    model_params: Optional[Mapping[str, Any]] = None,
    *,
    keys_order: Optional[Sequence[str]] = None,
) -> str:
    """Genera el nombre de archivo para un checkpoint de modelo.

    Args:
        model_id: Identificador base del modelo.
        model_params: Parámetros incluidos como tokens en el nombre.
        keys_order: Orden preferido para las claves del nombre.

    Returns:
        Nombre del checkpoint con extensión ``.pt``.
    """
    model_id = ensure_model_prefix(model_id)
    tokens = tokens_from_params(model_params or {}, keys_order=keys_order)
    base = f"{model_id}_{'_'.join(tokens)}" if tokens else f"{model_id}_"

    return ensure_pt_suffix(slug(base))


def build_history_file_name(run_tag: str) -> str:
    """Genera el nombre estandarizado para un historial de entrenamiento."""
    return ensure_json_suffix(f"{slug(run_tag)}.history")


def run_tag_from_int(run_id: int) -> str:
    """Formatea un entero como etiqueta de run."""
    return f"run{int(run_id):0{RUN_ID_WIDTH}d}"