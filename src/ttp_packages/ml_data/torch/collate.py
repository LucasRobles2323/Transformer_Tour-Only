#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/torch/collate.py

"""Funciones de collate para batches homogéneos TTP.

Este módulo agrupa samples individuales con el mismo número de ciudades e ítems
en batches tensoriales listos para entrenamiento.
"""

from __future__ import annotations

from typing import Any, Dict, List

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


def collate_same_size(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agrupa samples homogéneos en un batch tensorial.

    Args:
        batch: Lista de samples individuales con estructura ``meta``, ``inputs``
            y ``teacher``.

    Returns:
        Batch con secciones ``meta``, ``inputs`` y ``teacher``.

    Raises:
        ValueError: Si los samples tienen distinto número de ciudades o ítems.
    """
    # Un batch regular solo puede apilar tensores si todos los samples comparten N y M.
    n_cities_values = [sample["meta"]["n_cities"] for sample in batch]
    m_items_values = [sample["meta"]["m_items"] for sample in batch]

    if len(set(n_cities_values)) != 1 or len(set(m_items_values)) != 1:
        error_msg = (
            f"Batch con tamaños distintos. "
            f"Ns={set(n_cities_values)} Ms={set(m_items_values)}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    batch_size = len(batch)

    def stack_inputs(key: str) -> torch.Tensor:
        """Apila una entrada de todos los samples del batch."""
        return torch.stack([sample["inputs"][key] for sample in batch], dim=0)

    def stack_teacher(key: str) -> torch.Tensor:
        """Apila un target de todos los samples del batch."""
        return torch.stack([sample["teacher"][key] for sample in batch], dim=0)

    # Notación de shapes: B=batch_size, N=n_cities, M=m_items.
    return {
        "meta": [sample["meta"] for sample in batch],
        "inputs": {
            KEY_COORDS_RAW: stack_inputs(KEY_COORDS_RAW),  # (B, N, 2) reales
            KEY_CAPACITY: stack_inputs(KEY_CAPACITY).view(batch_size, 1),  # (B, 1)

            KEY_ITEM_CITY: stack_inputs(KEY_ITEM_CITY),  # (B, M)
            KEY_ITEM_PROFIT: stack_inputs(KEY_ITEM_PROFIT),  # (B, M)
            KEY_ITEM_WEIGHT: stack_inputs(KEY_ITEM_WEIGHT),  # (B, M)

            KEY_MIN_SPEED: stack_inputs(KEY_MIN_SPEED).view(batch_size, 1),  # (B, 1)
            KEY_MAX_SPEED: stack_inputs(KEY_MAX_SPEED).view(batch_size, 1),  # (B, 1)
            KEY_RENT: stack_inputs(KEY_RENT).view(batch_size, 1),  # (B, 1)
        },
        "teacher": {
            KEY_TOUR_NEXT: stack_teacher(KEY_TOUR_NEXT),  # (B, N)
            KEY_PICKS: stack_teacher(KEY_PICKS),  # (B, M)
            KEY_PROFIT: torch.cat(
                [sample["teacher"][KEY_PROFIT] for sample in batch],
                dim=0,
            ),  # (B,)
            KEY_TIME: torch.cat(
                [sample["teacher"][KEY_TIME] for sample in batch],
                dim=0,
            ),  # (B,)
            KEY_OBJECTIVE: torch.cat(
                [sample["teacher"][KEY_OBJECTIVE] for sample in batch],
                dim=0,
            ),  # (B,)
        },
    }