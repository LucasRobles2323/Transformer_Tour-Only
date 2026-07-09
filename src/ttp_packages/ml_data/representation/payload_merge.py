#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/representation/payload_merge.py

"""Unión de payloads tensoriales TTP.

Este módulo contiene lógica pura para unir datasets ya cargados en memoria.
No conoce rutas ni archivos; solo trabaja con payloads tensoriales que cumplen
el esquema compacto definido en ``ml_data.representation.payload``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import torch

from src.ttp_packages.ml_data.representation.payload import (
    INPUT_KEYS,
    TEACH_KEYS,
    validate_payload,
)


def _resolve_source_names(
    source_names: Optional[Sequence[str]],
    n_payloads: int,
) -> list[str]:
    """Normaliza los nombres de origen usados para trazabilidad.

    Args:
        source_names: Nombres opcionales asociados a cada payload.
        n_payloads: Cantidad esperada de payloads.

    Returns:
        Lista de nombres de origen, uno por payload.

    Raises:
        ValueError: Si ``source_names`` no tiene la misma longitud que los
            payloads.
    """
    if source_names is None:
        return [f"payload_{index}" for index in range(n_payloads)]

    if len(source_names) != n_payloads:
        raise ValueError(
            "source_names debe tener la misma cantidad de elementos que payloads. "
            f"Recibido {len(source_names)} para {n_payloads} payloads."
        )

    return [str(name) for name in source_names]


def _names_for_payload(
    payload: Mapping[str, Any],
    *,
    source_name: str,
    n_samples: int,
) -> list[str]:
    """Obtiene nombres de samples para un payload.

    Si el payload trae ``names`` con largo correcto, se reutilizan. Si faltan,
    están vacíos o tienen largo incompatible, se generan nombres trazables a
    partir del archivo/origen.

    Args:
        payload: Payload tensorial validado.
        source_name: Nombre del dataset de origen.
        n_samples: Cantidad de samples del payload.

    Returns:
        Lista de nombres de largo ``n_samples``.
    """
    raw_names = payload.get("names")

    if (
        isinstance(raw_names, Sequence)
        and not isinstance(raw_names, (str, bytes))
        and len(raw_names) == n_samples
    ):
        names: list[str] = []

        for sample_index, raw_name in enumerate(raw_names):
            clean_name = str(raw_name).strip()
            if clean_name:
                names.append(clean_name)
            else:
                names.append(f"{source_name}::sample_{sample_index}")

        return names

    return [
        f"{source_name}::sample_{sample_index}"
        for sample_index in range(n_samples)
    ]


def _tensor_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Mueve un tensor al dispositivo destino si hace falta.

    Args:
        tensor: Tensor de entrada.
        device: Dispositivo destino.

    Returns:
        Tensor ubicado en ``device``.
    """
    if tensor.device == device:
        return tensor

    return tensor.to(device)


def merge_tensor_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    source_names: Optional[Sequence[str]] = None,
    keep_names: bool = True,
) -> Dict[str, Any]:
    """Une múltiples payloads tensoriales TTP en un único payload.

    La unión concatena todos los tensores por la dimensión de samples
    ``dim=0``. Todos los payloads deben compartir el mismo número de ciudades
    ``N`` y de ítems ``M``.

    Args:
        payloads: Secuencia de payloads a unir.
        source_names: Nombres opcionales de origen, normalmente nombres de
            archivo. Se usan para generar ``names`` cuando algún payload no los
            trae.
        keep_names: Si es ``True``, agrega ``payload["names"]`` al resultado.

    Returns:
        Payload tensorial consolidado y validado.

    Raises:
        ValueError: Si no hay payloads, si algún payload es inválido o si las
            dimensiones ``N``/``M`` no coinciden.
    """
    if not payloads:
        raise ValueError("No hay payloads para unir.")

    payload_list = list(payloads)
    resolved_source_names = _resolve_source_names(source_names, len(payload_list))

    first_samples, first_n_cities, first_m_items = validate_payload(
        dict(payload_list[0])
    )

    # Se usa el dispositivo del primer payload como destino de concatenación.
    ref_key = INPUT_KEYS[0]
    target_device = payload_list[0]["inputs"][ref_key].device

    input_tensors: dict[str, list[torch.Tensor]] = {
        key: [] for key in INPUT_KEYS
    }
    teacher_tensors: dict[str, list[torch.Tensor]] = {
        key: [] for key in TEACH_KEYS
    }

    merged_names: list[str] = []
    total_samples = 0

    for payload_index, payload in enumerate(payload_list):
        n_samples, n_cities, m_items = validate_payload(dict(payload))

        if n_cities != first_n_cities or m_items != first_m_items:
            source_name = resolved_source_names[payload_index]
            raise ValueError(
                f"Dataset incompatible en '{source_name}': "
                f"(N,M)=({n_cities},{m_items}) != "
                f"({first_n_cities},{first_m_items})."
            )

        total_samples += int(n_samples)

        for key in INPUT_KEYS:
            tensor = payload["inputs"][key]
            input_tensors[key].append(_tensor_on_device(tensor, target_device))

        for key in TEACH_KEYS:
            tensor = payload["teacher"][key]
            teacher_tensors[key].append(_tensor_on_device(tensor, target_device))

        if keep_names:
            merged_names.extend(
                _names_for_payload(
                    payload,
                    source_name=resolved_source_names[payload_index],
                    n_samples=int(n_samples),
                )
            )

    merged_payload: Dict[str, Any] = {
        "n_cities": int(first_n_cities),
        "m_items": int(first_m_items),
        "num_samples": int(total_samples),
        "inputs": {},
        "teacher": {},
    }

    for key in INPUT_KEYS:
        # La dimensión 0 representa samples; el resto del shape se mantiene.
        merged_payload["inputs"][key] = torch.cat(
            input_tensors[key],
            dim=0,
        ).contiguous()

    for key in TEACH_KEYS:
        merged_payload["teacher"][key] = torch.cat(
            teacher_tensors[key],
            dim=0,
        ).contiguous()

    if keep_names:
        merged_payload["names"] = merged_names

    n_final_samples, _, _ = validate_payload(merged_payload)
    merged_payload["num_samples"] = int(n_final_samples)

    return merged_payload