#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/log_format.py

"""Formateo legible de estructuras para log.

Este módulo centraliza helpers para registrar diccionarios, listas y dataclasses
en múltiples líneas. Evita logs ilegibles cuando se imprimen configuraciones,
summaries o parámetros anidados.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


_SIMPLE_TYPES = (str, int, float, bool, type(None))


def to_loggable(value: Any) -> Any:
    """Convierte un objeto a una estructura serializable para log.

    Args:
        value: Objeto arbitrario a convertir.

    Returns:
        Versión compuesta por tipos simples, listas y diccionarios.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return to_loggable(asdict(value))

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): to_loggable(item_value)
            for key, item_value in value.items()
        }

    if isinstance(value, tuple):
        return [to_loggable(item) for item in value]

    if isinstance(value, list):
        return [to_loggable(item) for item in value]

    if isinstance(value, set):
        return sorted(to_loggable(item) for item in value)

    # Soporte ligero para escalares NumPy/PyTorch sin importar esas librerías.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass

    return value


def _is_simple_value(value: Any) -> bool:
    """Indica si un valor puede imprimirse en una sola línea.

    Args:
        value: Valor a evaluar.

    Returns:
        ``True`` si es un valor simple; ``False`` si es anidado.
    """
    return isinstance(value, _SIMPLE_TYPES)


def _indent_lines(text: str, spaces: int) -> str:
    """Indenta todas las líneas de un texto.

    Args:
        text: Texto a indentar.
        spaces: Cantidad de espacios a anteponer.

    Returns:
        Texto indentado.
    """
    prefix = " " * int(spaces)
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def format_pretty_value(
    value: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> str:
    """Formatea un valor potencialmente anidado en varias líneas.

    Args:
        value: Valor a formatear.
        indent: Indentación interna usada por JSON.
        sort_keys: Si es ``True``, ordena claves alfabéticamente.

    Returns:
        Texto legible para terminal y archivos de log.
    """
    plain_value = to_loggable(value)

    if _is_simple_value(plain_value):
        return str(plain_value)

    return json.dumps(
        plain_value,
        indent=int(indent),
        ensure_ascii=False,
        sort_keys=bool(sort_keys),
        default=str,
    )


def format_key_value_block(
    title: Optional[str],
    values: Mapping[str, Any],
    *,
    indent: int = 2,
    nested_indent: int = 4,
    key_order: Optional[Sequence[str]] = None,
    skip_none: bool = False,
    sort_nested_keys: bool = False,
) -> str:
    """Construye un bloque legible de claves y valores.

    Los valores simples quedan en una línea. Los valores anidados se imprimen
    debajo de su clave usando JSON indentado.

    Args:
        title: Título opcional del bloque.
        values: Diccionario a imprimir.
        indent: Indentación para las claves principales.
        nested_indent: Indentación para estructuras anidadas.
        key_order: Orden preferido de claves. Las claves no incluidas se agregan
            al final en el orden original.
        skip_none: Si es ``True``, omite claves con valor ``None``.
        sort_nested_keys: Si es ``True``, ordena claves dentro de objetos JSON.

    Returns:
        Bloque de texto listo para log.
    """
    lines: list[str] = []

    if title:
        lines.append(str(title))

    used_keys: set[str] = set()
    ordered_keys: list[str] = []

    if key_order is not None:
        for key in key_order:
            if key in values:
                ordered_keys.append(str(key))
                used_keys.add(str(key))

    for key in values.keys():
        str_key = str(key)
        if str_key not in used_keys:
            ordered_keys.append(str_key)

    key_prefix = " " * int(indent)

    for key in ordered_keys:
        value = values[key]

        if skip_none and value is None:
            continue

        plain_value = to_loggable(value)

        if _is_simple_value(plain_value):
            lines.append(f"{key_prefix}{key}: {plain_value}")
            continue

        lines.append(f"{key_prefix}{key}:")
        pretty_value = format_pretty_value(
            plain_value,
            indent=2,
            sort_keys=sort_nested_keys,
        )
        lines.append(_indent_lines(pretty_value, nested_indent))

    return "\n".join(lines)


def log_block(
    log_fn: Callable[[str], None],
    block: str,
) -> None:
    """Registra un bloque multilínea línea por línea.

    Args:
        log_fn: Función de log.
        block: Texto posiblemente multilínea.
    """
    for line in str(block).splitlines():
        log_fn(line)


def log_key_value_block(
    log_fn: Callable[[str], None],
    title: Optional[str],
    values: Mapping[str, Any],
    *,
    indent: int = 2,
    nested_indent: int = 4,
    key_order: Optional[Sequence[str]] = None,
    skip_none: bool = False,
    sort_nested_keys: bool = False,
) -> None:
    """Registra un diccionario en formato multilínea legible.

    Args:
        log_fn: Función de log.
        title: Título opcional del bloque.
        values: Diccionario a registrar.
        indent: Indentación para claves principales.
        nested_indent: Indentación para estructuras anidadas.
        key_order: Orden preferido de claves.
        skip_none: Si es ``True``, omite claves con valor ``None``.
        sort_nested_keys: Si es ``True``, ordena claves dentro de objetos JSON.
    """
    block = format_key_value_block(
        title,
        values,
        indent=indent,
        nested_indent=nested_indent,
        key_order=key_order,
        skip_none=skip_none,
        sort_nested_keys=sort_nested_keys,
    )
    log_block(log_fn, block)