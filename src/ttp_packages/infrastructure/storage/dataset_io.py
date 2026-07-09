#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/dataset_io.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.ml_data.representation.payload import (
    INPUT_KEYS, TEACH_KEYS,
    validate_payload,
    samples_to_tensor_payload,
    empty_payload,
)
from src.ttp_packages.ml_data.representation.payload_merge import (
    merge_tensor_payloads,
)

from .paths import build_dataset_path
from .torch_io import atomic_torch_save, load_torch_dict


# Inicialización del logger para este módulo específico
logger = setup_logger(__name__)


def _validate_dataset_dimensions(
    payload: Dict[str, Any],
    *,
    n_cities: int,
    m_items: int,
    file_name: str,
) -> None:
    """Valida que un payload tenga las dimensiones esperadas.

    Args:
        payload: Dataset cargado o creado.
        n_cities: Número esperado de ciudades.
        m_items: Número esperado de ítems.
        file_name: Nombre del archivo asociado, usado para mensajes de error.

    Raises:
        ValueError: Si las dimensiones del payload no coinciden.
    """
    payload_n = int(payload["n_cities"])
    payload_m = int(payload["m_items"])

    if payload_n != int(n_cities) or payload_m != int(m_items):
        raise ValueError(
            f"Dataset '{file_name}' tiene dimensiones incompatibles. "
            f"Archivo (N,M)=({payload_n},{payload_m}) "
            f"pedido (N,M)=({n_cities},{m_items})."
        )

def load_dataset(
    file_name: str,
    verbose: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Carga un dataset PyTorch desde disco.

    Args:
        file_name: Nombre del archivo dentro del directorio de datasets.
        verbose: Si es True, registra información de carga.
        map_location: Dispositivo destino para ``torch.load``. Si es ``None``,
            carga en CPU.
        log_fn: Función opcional de logging.

    Returns:
        Payload validado del dataset, incluyendo ``num_samples``.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el payload cargado no cumple el esquema esperado.
    """
    if log_fn is None:
        log_fn = logger.info

    path = build_dataset_path(file_name)

    if not path.exists():
        raise FileNotFoundError(f"No existe dataset: {path}")

    if map_location is None:
        map_location = "cpu"

    payload = load_torch_dict(path, map_location=map_location)
    n_samples, n_cities, n_items = validate_payload(payload)

    payload["num_samples"] = int(n_samples)

    if verbose:
        log_fn(
            f"\n[DATASET LOAD] {file_name} -> "
            f"total samples={n_samples} N={n_cities} M={n_items}\n"
        )

    return payload


def save_dataset(
    file_name: str,
    payload: Dict[str, Any],
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    """Guarda un dataset PyTorch en disco.

    Args:
        file_name: Nombre del archivo destino.
        payload: Dataset a guardar.
        verbose: Si es True, registra información del guardado.
        log_fn: Función opcional de logging.

    Returns:
        Ruta absoluta donde se guardó el archivo.

    Raises:
        ValueError: Si el payload no cumple el esquema esperado.
    """
    if log_fn is None:
        log_fn = logger.info

    path = build_dataset_path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_samples, n_cities, n_items = validate_payload(payload)
    payload["num_samples"] = int(n_samples)

    atomic_torch_save(payload, path)

    if verbose:
        log_fn(
            f"\n[DATASET SAVE] {file_name} -> "
            f"total samples={n_samples} N={n_cities} M={n_items}\n"
        )

    return path


def prepare_dataset_for_append(
    file_name: str,
    n_cities: int,
    m_items: int,
    verbose: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
    save_empty: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], int]:
    """Crea un dataset vacío o carga uno existente para expandirlo.

    Si el archivo ya existe, no se sobrescribe. En ese caso se registra una
    alerta, se carga el dataset existente y se validan sus dimensiones.

    Args:
        file_name: Nombre del archivo del dataset.
        n_cities: Número esperado de ciudades.
        m_items: Número esperado de ítems.
        verbose: Si es True, registra información del proceso.
        map_location: Dispositivo destino al cargar un dataset existente.
        save_empty: Si es True, guarda inmediatamente el dataset vacío cuando
            el archivo no existe.
        log_fn: Función opcional de logging.

    Returns:
        Tupla ``(payload, num_samples)``.

    Raises:
        ValueError: Si el archivo existe pero sus dimensiones no coinciden.
    """
    if log_fn is None:
        log_fn = logger.info

    if map_location is None:
        map_location = "cpu"

    path = build_dataset_path(file_name)

    if path.exists():
        if verbose:
            log_fn(
                "[DATASET EXISTS] %s ya existe. Se cargará para expandirlo.",
                file_name,
            )

        payload = load_dataset(
            file_name,
            verbose=verbose,
            map_location=map_location,
            log_fn=log_fn,
        )
        _validate_dataset_dimensions(
            payload,
            n_cities=n_cities,
            m_items=m_items,
            file_name=file_name,
        )
        return payload, int(payload["num_samples"])

    payload = empty_payload(n_cities=n_cities, m_items=m_items)
    payload["num_samples"] = 0

    if save_empty:
        save_dataset(
            file_name,
            payload,
            verbose=verbose,
            log_fn=log_fn,
        )
    elif verbose:
        log_fn(
            f"[DATASET CREATE] {file_name} -> "
            f"total samples=0 N={n_cities} M={m_items}"
        )

    return payload, 0


def append_samples_to_payload(
    payload: Dict[str, Any],
    new_samples: List[Dict[str, Any]],
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Añade muestras nuevas a un payload existente.

    Convierte las muestras nuevas a tensores, valida dimensiones y concatena los
    tensores sobre la dimensión de muestras.

    Args:
        payload: Dataset acumulado.
        new_samples: Muestras nuevas en formato compacto.
        verbose: Si es True, registra información del append.
        log_fn: Función opcional de logging.

    Returns:
        Payload actualizado.

    Raises:
        ValueError: Si las dimensiones ``N`` o ``M`` no coinciden.
    """
    if not new_samples:
        return payload

    if log_fn is None:
        log_fn = logger.info

    _, n_cities, n_items = validate_payload(payload)

    new_payload = samples_to_tensor_payload(
        new_samples,
        store_names=("names" in payload),
    )
    n_new_samples, new_n_cities, new_n_items = validate_payload(new_payload)

    if n_cities != new_n_cities or n_items != new_n_items:
        raise ValueError(
            f"Append inválido: viejo (N,M)=({n_cities},{n_items}) "
            f"nuevo (N,M)=({new_n_cities},{new_n_items})"
        )

    # Alinear device del payload nuevo con el payload existente antes de concatenar.
    ref_input_key = INPUT_KEYS[0]
    target_device = payload["inputs"][ref_input_key].device

    for key in INPUT_KEYS:
        if torch.is_tensor(new_payload["inputs"][key]):
            new_payload["inputs"][key] = new_payload["inputs"][key].to(target_device)

    for key in TEACH_KEYS:
        if torch.is_tensor(new_payload["teacher"][key]):
            new_payload["teacher"][key] = new_payload["teacher"][key].to(target_device)

    for key in INPUT_KEYS:
        payload["inputs"][key] = torch.cat(
            [payload["inputs"][key], new_payload["inputs"][key]],
            dim=0,
        ).contiguous()

    for key in TEACH_KEYS:
        payload["teacher"][key] = torch.cat(
            [payload["teacher"][key], new_payload["teacher"][key]],
            dim=0,
        ).contiguous()

    if "names" in payload and "names" in new_payload:
        payload["names"] = list(payload["names"]) + list(new_payload["names"])

    n_final_samples, _, _ = validate_payload(payload)
    payload["num_samples"] = int(n_final_samples)

    if verbose:
        log_fn(
            f"[DATASET APPEND] +{int(n_new_samples)} -> "
            f"total samples={int(n_final_samples)}"
        )

    return payload


def add_samples_to_file(
    file_name: str,
    n_cities: int,
    m_items: int,
    samples: List[Dict[str, Any]],
    verbose: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    """Carga o crea un dataset, agrega muestras y guarda el resultado.

    Args:
        file_name: Nombre del archivo del dataset.
        n_cities: Número esperado de ciudades.
        m_items: Número esperado de ítems.
        samples: Lista de nuevas muestras a insertar.
        verbose: Si es True, registra información del proceso.
        map_location: Dispositivo destino al cargar un dataset existente.
        log_fn: Función opcional de logging.

    Returns:
        Ruta absoluta donde se guardó el dataset actualizado.

    Raises:
        ValueError: Si las dimensiones del dataset existente no coinciden.
    """
    if map_location is None:
        map_location = "cpu"

    payload, _ = prepare_dataset_for_append(
        file_name,
        n_cities=n_cities,
        m_items=m_items,
        verbose=verbose,
        map_location=map_location,
        save_empty=False,
        log_fn=log_fn,
    )

    payload = append_samples_to_payload(
        payload,
        samples,
        verbose=verbose,
        log_fn=log_fn,
    )

    return save_dataset(file_name, payload, verbose=verbose, log_fn=log_fn)


def _validate_merge_file_names(
    input_file_names: Sequence[str],
    output_file_name: str,
) -> None:
    """Valida nombres de archivos antes de unir datasets.

    Args:
        input_file_names: Archivos de entrada.
        output_file_name: Archivo de salida.

    Raises:
        ValueError: Si no hay inputs, si hay nombres vacíos, si hay duplicados
            o si el output coincide con algún input.
    """
    if not input_file_names:
        raise ValueError("Debes entregar al menos un dataset de entrada.")

    cleaned_inputs = [str(file_name).strip() for file_name in input_file_names]
    cleaned_output = str(output_file_name).strip()

    if not cleaned_output:
        raise ValueError("output_file_name no puede estar vacío.")

    for file_name in cleaned_inputs:
        if not file_name:
            raise ValueError("input_file_names contiene un nombre vacío.")

    duplicated = {
        file_name
        for file_name in cleaned_inputs
        if cleaned_inputs.count(file_name) > 1
    }
    if duplicated:
        raise ValueError(
            "Hay datasets de entrada repetidos. "
            f"Duplicados: {sorted(duplicated)}"
        )

    input_paths = {
        build_dataset_path(file_name).resolve()
        for file_name in cleaned_inputs
    }
    output_path = build_dataset_path(cleaned_output).resolve()

    if output_path in input_paths:
        raise ValueError(
            "El dataset de salida no puede ser también un dataset de entrada."
        )


def merge_dataset_files(
    input_file_names: Sequence[str],
    output_file_name: str,
    *,
    overwrite: bool = False,
    keep_names: bool = True,
    verbose: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], Path, Dict[str, int]]:
    """Une múltiples archivos de dataset ``.pt`` en un único dataset.

    Carga cada dataset, valida que todos compartan dimensiones ``N`` y ``M``,
    concatena sus samples y guarda el resultado en ``output_file_name``.

    Args:
        input_file_names: Nombres de datasets existentes dentro del directorio
            configurado para datasets.
        output_file_name: Nombre del dataset unido a crear.
        overwrite: Si es ``True``, permite sobrescribir el archivo de salida.
        keep_names: Si es ``True``, preserva o genera ``payload["names"]``.
        verbose: Si es ``True``, registra progreso de carga, merge y guardado.
        map_location: Dispositivo usado al cargar los datasets. Si es ``None``,
            carga en CPU.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Tupla ``(payload_unido, ruta_salida, samples_por_input)``.

    Raises:
        FileExistsError: Si el output existe y ``overwrite`` es ``False``.
        FileNotFoundError: Si algún input no existe.
        ValueError: Si algún payload es inválido o incompatible.
    """
    if log_fn is None:
        log_fn = logger.info

    if map_location is None:
        map_location = "cpu"

    _validate_merge_file_names(
        input_file_names=input_file_names,
        output_file_name=output_file_name,
    )

    output_path = build_dataset_path(output_file_name)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"El dataset de salida ya existe: {output_path}. "
            "Usa overwrite=True si quieres reemplazarlo."
        )

    payloads: List[Dict[str, Any]] = []
    input_num_samples: Dict[str, int] = {}

    if verbose:
        log_fn(
            f"\n[DATASET MERGE] Cargando {len(input_file_names)} datasets..."
        )

    for file_name in input_file_names:
        payload = load_dataset(
            file_name,
            verbose=verbose,
            map_location=map_location,
            log_fn=log_fn,
        )
        n_samples, _, _ = validate_payload(payload)

        payloads.append(payload)
        input_num_samples[str(file_name)] = int(n_samples)

    merged_payload = merge_tensor_payloads(
        payloads,
        source_names=[str(file_name) for file_name in input_file_names],
        keep_names=keep_names,
    )

    n_samples, n_cities, m_items = validate_payload(merged_payload)

    if verbose:
        log_fn("\n[DATASET MERGE] Resumen de entrada:")
        for file_name, samples in input_num_samples.items():
            log_fn(f"  - {file_name}: {samples} samples")

        log_fn(
            "[DATASET MERGE] Resultado: "
            f"total samples={n_samples} N={n_cities} M={m_items}"
        )

    saved_path = save_dataset(
        output_file_name,
        merged_payload,
        verbose=verbose,
        log_fn=log_fn,
    )

    return merged_payload, saved_path, input_num_samples
