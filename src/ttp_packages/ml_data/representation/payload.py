#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/representation/payload.py

"""Construcción y validación de payloads tensoriales TTP.

Este módulo define el esquema compacto de datasets TTP, valida shapes y dtypes,
convierte listas de samples a tensores apilados y crea payloads vacíos.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.ml_data.config import (
    KEY_CAPACITY,
    KEY_COORDS_RAW,
    KEY_ITEM_CITY,
    KEY_ITEM_PROFIT,
    KEY_ITEM_WEIGHT,
    KEY_MAX_SPEED,
    KEY_MIN_SPEED,
    KEY_OBJECTIVE,
    KEY_PICKS,
    KEY_PROFIT,
    KEY_RENT,
    KEY_TIME,
    KEY_TOUR_NEXT,
)


logger = setup_logger(__name__)

# =============================================================================
# Esquema único compacto (tensores apilados, homogéneo N/M, sin matrices N×N)
# =============================================================================
INPUT_KEYS = [
    KEY_COORDS_RAW,
    KEY_CAPACITY,
    KEY_ITEM_CITY,
    KEY_ITEM_PROFIT,
    KEY_ITEM_WEIGHT,
    KEY_MIN_SPEED,
    KEY_MAX_SPEED,
    KEY_RENT,
]

TEACH_KEYS = [
    KEY_TOUR_NEXT,
    KEY_PICKS,
    KEY_PROFIT,
    KEY_TIME,
    KEY_OBJECTIVE,
]


def _require(cond: bool, msg: str) -> None:
    """Valida una condición y lanza ``ValueError`` si falla.

    Args:
        cond: Condición a evaluar.
        msg: Mensaje usado para log y excepción.

    Raises:
        ValueError: Si ``cond`` es False.
    """
    if not cond:
        logger.error(msg)
        raise ValueError(msg)


def validate_payload(payload: Dict[str, Any]) -> Tuple[int, int, int]:
    """Valida estructura, shapes y dtypes de un payload TTP.

    Args:
        payload: Payload tensorial a validar.

    Returns:
        Tupla ``(S, N, M)``, donde ``S`` es el número de samples,
        ``N`` el número de ciudades y ``M`` el número de ítems.

    Raises:
        ValueError: Si falta alguna clave, shape o dtype esperado.
    """
    _require(isinstance(payload, dict), "Payload inválido: no es dict.")
    _require(
        "n_cities" in payload and "m_items" in payload,
        "Payload inválido: falta n_cities/m_items.",
    )
    _require(
        "inputs" in payload and "teacher" in payload,
        "Payload inválido: falta inputs/teacher.",
    )

    n_cities = int(payload["n_cities"])
    m_items = int(payload["m_items"])
    inputs_section = payload["inputs"]
    teacher_section = payload["teacher"]

    _require(isinstance(inputs_section, dict), "Payload inválido: inputs no es dict.")
    _require(
        isinstance(teacher_section, dict),
        "Payload inválido: teacher no es dict.",
    )

    for key in INPUT_KEYS:
        _require(
            key in inputs_section,
            f"Payload inválido: falta inputs['{key}'].",
        )

    for key in TEACH_KEYS:
        _require(
            key in teacher_section,
            f"Payload inválido: falta teacher['{key}'].",
        )

    coords_raw = inputs_section[KEY_COORDS_RAW]
    _require(
        isinstance(coords_raw, torch.Tensor),
        f"inputs['{KEY_COORDS_RAW}'] debe ser Tensor.",
    )
    _require(
        (
            coords_raw.ndim == 3
            and coords_raw.shape[1] == n_cities
            and coords_raw.shape[2] == 2
        ),
        f"inputs['{KEY_COORDS_RAW}'] shape inválido. "
        f"Esperado (S,{n_cities},2), recibido {tuple(coords_raw.shape)}.",
    )

    n_samples = int(coords_raw.shape[0])

    # Todas las entradas y targets deben compartir la misma dimensión de batch S.
    def _check(
        tensor: torch.Tensor,
        name: str,
        shape_tail: Tuple[int, ...],
        dtype: torch.dtype | None = None,
    ) -> None:
        """Valida shape y dtype de un tensor del payload.

        Args:
            tensor: Tensor a validar.
            name: Nombre descriptivo usado en mensajes de error.
            shape_tail: Shape esperado después de la dimensión de batch.
            dtype: Dtype esperado. Si es ``None``, no se valida dtype.

        Raises:
            ValueError: Si el tensor no cumple shape o dtype esperado.
        """
        _require(isinstance(tensor, torch.Tensor), f"{name} debe ser Tensor.")
        _require(
            int(tensor.shape[0]) == n_samples,
            f"{name} dim0 inválida: esperado S={n_samples}, "
            f"recibido {int(tensor.shape[0])}.",
        )
        _require(
            tuple(tensor.shape[1:]) == shape_tail,
            f"{name} shape inválido. "
            f"Esperado (S,{','.join(map(str, shape_tail))}), "
            f"recibido {tuple(tensor.shape)}.",
        )

        if dtype is not None:
            _require(
                tensor.dtype == dtype,
                f"{name} dtype inválido. Esperado {dtype}, recibido {tensor.dtype}.",
            )

    _check(inputs_section[KEY_CAPACITY], f"inputs['{KEY_CAPACITY}']", (1,))
    _check(
        inputs_section[KEY_ITEM_CITY],
        f"inputs['{KEY_ITEM_CITY}']",
        (m_items,),
        dtype=torch.long,
    )
    _check(
        inputs_section[KEY_ITEM_PROFIT],
        f"inputs['{KEY_ITEM_PROFIT}']",
        (m_items,),
    )
    _check(
        inputs_section[KEY_ITEM_WEIGHT],
        f"inputs['{KEY_ITEM_WEIGHT}']",
        (m_items,),
    )
    _check(inputs_section[KEY_MIN_SPEED], f"inputs['{KEY_MIN_SPEED}']", (1,))
    _check(inputs_section[KEY_MAX_SPEED], f"inputs['{KEY_MAX_SPEED}']", (1,))
    _check(inputs_section[KEY_RENT], f"inputs['{KEY_RENT}']", (1,))

    _check(
        teacher_section[KEY_TOUR_NEXT],
        f"teacher['{KEY_TOUR_NEXT}']",
        (n_cities,),
        dtype=torch.long,
    )
    _check(
        teacher_section[KEY_PICKS],
        f"teacher['{KEY_PICKS}']",
        (m_items,),
    )

    _check(teacher_section[KEY_PROFIT], f"teacher['{KEY_PROFIT}']", tuple())
    _check(teacher_section[KEY_TIME], f"teacher['{KEY_TIME}']", tuple())
    _check(
        teacher_section[KEY_OBJECTIVE],
        f"teacher['{KEY_OBJECTIVE}']",
        tuple(),
    )

    return n_samples, n_cities, m_items


def ensure_homogeneous_samples(samples: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Verifica que todos los samples tengan el mismo tamaño.

    Args:
        samples: Lista de samples compactos.

    Returns:
        Tupla ``(N, M)`` compartida por todos los samples.

    Raises:
        ValueError: Si la lista está vacía o si algún sample difiere en
            ``N`` o ``M`` respecto al primer elemento.
    """
    _require(len(samples) > 0, "samples vacío.")

    n_cities = int(samples[0]["meta"]["n_cities"])
    m_items = int(samples[0]["meta"]["m_items"])

    for index, sample in enumerate(samples):
        sample_n_cities = int(sample["meta"]["n_cities"])
        sample_m_items = int(sample["meta"]["m_items"])

        if sample_n_cities != n_cities or sample_m_items != m_items:
            error_msg = (
                f"Samples NO homogéneos en i={index}: "
                f"(N,M)=({sample_n_cities},{sample_m_items}) "
                f"!= ({n_cities},{m_items})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    return n_cities, m_items


def samples_to_tensor_payload(
    samples: List[Dict[str, Any]],
    store_names: bool = True,
) -> Dict[str, Any]:
    """Convierte una lista de samples en un payload tensorial compacto.

    Apila los tensores de cada sample individual en una dimensión extra de batch
    ``dim=0``. Requiere que todos los samples sean homogéneos en tamaño ``N`` y
    ``M``.

    Args:
        samples: Lista de samples a procesar.
        store_names: Si es True, guarda los nombres de las instancias en
            ``payload["names"]``.

    Returns:
        Payload consolidado válido según ``validate_payload``.
    """
    n_cities, m_items = ensure_homogeneous_samples(samples)
    n_samples = len(samples)

    inputs: Dict[str, torch.Tensor] = {}

    for key in INPUT_KEYS:
        tensors = [samples[index]["inputs"][key] for index in range(n_samples)]

        # KEY_ITEM_CITY debe conservar dtype long porque se usa como índice de ciudad.
        if key == KEY_ITEM_CITY:
            tensors = [tensor.long() for tensor in tensors]
        else:
            tensors = [tensor.float() for tensor in tensors]

        inputs[key] = torch.stack(tensors, dim=0).contiguous()

    teacher: Dict[str, torch.Tensor] = {
        KEY_TOUR_NEXT: torch.stack(
            [
                samples[index]["teacher"][KEY_TOUR_NEXT].long()
                for index in range(n_samples)
            ],
            dim=0,
        ).contiguous(),
        KEY_PICKS: torch.stack(
            [
                samples[index]["teacher"][KEY_PICKS].float()
                for index in range(n_samples)
            ],
            dim=0,
        ).contiguous(),
        KEY_PROFIT: torch.tensor(
            [
                float(samples[index]["teacher"][KEY_PROFIT])
                for index in range(n_samples)
            ],
            dtype=torch.float32,
        ),
        KEY_TIME: torch.tensor(
            [
                float(samples[index]["teacher"][KEY_TIME])
                for index in range(n_samples)
            ],
            dtype=torch.float32,
        ),
        KEY_OBJECTIVE: torch.tensor(
            [
                float(samples[index]["teacher"][KEY_OBJECTIVE])
                for index in range(n_samples)
            ],
            dtype=torch.float32,
        ),
    }

    payload: Dict[str, Any] = {
        "n_cities": int(n_cities),
        "m_items": int(m_items),
        "num_samples": int(n_samples),
        "inputs": inputs,
        "teacher": teacher,
    }

    if store_names:
        payload["names"] = [
            str(sample["meta"].get("name", ""))
            for sample in samples
        ]

    validate_payload(payload)
    return payload


def empty_payload(n_cities: int, m_items: int) -> Dict[str, Any]:
    """Crea un payload vacío con tensores inicializados.

    El payload mantiene ``S=0`` y conserva el mismo esquema compacto usado por
    datasets no vacíos.

    Args:
        n_cities: Número de ciudades esperadas.
        m_items: Número de ítems esperados.

    Returns:
        Payload válido y estructurado con ``num_samples=0``.
    """
    n_cities = int(n_cities)
    m_items = int(m_items)

    payload: Dict[str, Any] = {
        "n_cities": n_cities,
        "m_items": m_items,
        "num_samples": 0,
        "inputs": {
            KEY_COORDS_RAW: torch.empty((0, n_cities, 2), dtype=torch.float32),
            KEY_CAPACITY: torch.empty((0, 1), dtype=torch.float32),
            KEY_ITEM_CITY: torch.empty((0, m_items), dtype=torch.long),
            KEY_ITEM_PROFIT: torch.empty((0, m_items), dtype=torch.float32),
            KEY_ITEM_WEIGHT: torch.empty((0, m_items), dtype=torch.float32),
            KEY_MIN_SPEED: torch.empty((0, 1), dtype=torch.float32),
            KEY_MAX_SPEED: torch.empty((0, 1), dtype=torch.float32),
            KEY_RENT: torch.empty((0, 1), dtype=torch.float32),
        },
        "teacher": {
            KEY_TOUR_NEXT: torch.empty((0, n_cities), dtype=torch.long),
            KEY_PICKS: torch.empty((0, m_items), dtype=torch.float32),
            KEY_PROFIT: torch.empty((0,), dtype=torch.float32),
            KEY_TIME: torch.empty((0,), dtype=torch.float32),
            KEY_OBJECTIVE: torch.empty((0,), dtype=torch.float32),
        },
        "names": [],
    }

    validate_payload(payload)
    return payload