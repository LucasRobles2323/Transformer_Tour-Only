#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/generation/math/item_sampling.py

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from src.ttp_packages.generation.config import (
    DEFAULT_ITEM_MAX_VAL,
    DEFAULT_ITEM_MIN_VAL,
    SIMILAR_WEIGHTS_MAX,
    SIMILAR_WEIGHTS_MIN,
    STRONGLY_CORR_BONUS,
    CorrelationType,
)


def generate_items_logic(
    item_factor: int,
    corr_type: CorrelationType,
    rng: Optional[random.Random] = None,
) -> List[Tuple[int, int]]:
    """Genera pares ``(profit, weight)`` según una correlación dada.

    Args:
        item_factor: Cantidad de ítems a generar.
        corr_type: Tipo de correlación profit-peso.
        rng: Generador pseudoaleatorio opcional.

    Returns:
        Lista de pares ``(profit, weight)``.

    Raises:
        ValueError: Si ``item_factor`` es negativo o ``corr_type`` no está
            soportado.
    """
    if item_factor < 0:
        raise ValueError("item_factor no puede ser negativo.")

    if rng is None:
        rng = random.Random()

    items_data: List[Tuple[int, int]] = []

    for _ in range(item_factor):
        if corr_type == CorrelationType.UNCORRELATED:
            weight = rng.randint(DEFAULT_ITEM_MIN_VAL, DEFAULT_ITEM_MAX_VAL)
            profit = rng.randint(DEFAULT_ITEM_MIN_VAL, DEFAULT_ITEM_MAX_VAL)

        elif corr_type == CorrelationType.UNCORRELATED_SIMILAR_WEIGHTS:
            weight = rng.randint(SIMILAR_WEIGHTS_MIN, SIMILAR_WEIGHTS_MAX)
            profit = rng.randint(DEFAULT_ITEM_MIN_VAL, DEFAULT_ITEM_MAX_VAL)

        elif corr_type == CorrelationType.BOUNDED_STRONGLY_CORR:
            weight = rng.randint(DEFAULT_ITEM_MIN_VAL, DEFAULT_ITEM_MAX_VAL)
            profit = weight + STRONGLY_CORR_BONUS

        else:
            raise ValueError(f"Tipo de correlación desconocido: {corr_type}")

        items_data.append((profit, weight))

    return items_data