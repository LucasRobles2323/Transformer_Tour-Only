#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/torch/samplers.py

"""Samplers PyTorch para batches homogéneos por tamaño.

Este módulo agrupa índices de datasets TTP en buckets por ``(N, M)`` para evitar
batches con shapes incompatibles.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Tuple

from torch.utils.data import Dataset, Sampler


class BucketBatchSampler(Sampler[List[int]]):
    """Sampler personalizado que agrupa instancias de tamaños idénticos.

    Agrupa instancias en buckets que comparten exactamente el mismo número de
    ciudades ``N`` e ítems ``M``, asegurando que los batches emitidos sean
    homogéneos. Es útil si se combinan varios archivos heterogéneos en un solo
    dataset general.

    Attributes:
        dataset: Dataset original.
        batch_size: Cantidad de índices por batch.
        shuffle: Si es True, baraja buckets e índices.
        drop_last: Si es True, descarta batches incompletos.
        buckets: Mapa ``(N, M) -> índices``.
        keys: Lista de tamaños disponibles.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """Inicializa el sampler y construye buckets por tamaño.

        Args:
            dataset: Dataset cuyos elementos exponen ``meta["n_cities"]`` y
                ``meta["m_items"]``.
            batch_size: Tamaño de batch.
            shuffle: Si es True, baraja buckets e índices en cada iteración.
            drop_last: Si es True, descarta batches incompletos.

        Raises:
            ValueError: Si ``batch_size`` no es positivo.
        """
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size debe ser positivo.")

        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        # Cada bucket agrupa índices con shapes compatibles para collate_same_size().
        buckets: Dict[Tuple[int, int], List[int]] = {}

        for index in range(len(dataset)):
            sample = dataset[index]
            n_cities = int(sample["meta"]["n_cities"])
            m_items = int(sample["meta"]["m_items"])
            buckets.setdefault((n_cities, m_items), []).append(index)

        self.buckets = buckets
        self.keys = list(buckets.keys())

    def __iter__(self) -> Iterable[List[int]]:
        """Itera sobre batches homogéneos de índices.

        Yields:
            Batch de índices cuyos samples tienen el mismo ``N`` y ``M``.
        """
        keys = self.keys[:]
        if self.shuffle:
            random.shuffle(keys)

        for bucket_key in keys:
            indices = self.buckets[bucket_key][:]
            if self.shuffle:
                random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]

                if len(batch) < self.batch_size and self.drop_last:
                    continue

                yield batch

    def __len__(self) -> int:
        """Calcula el total de batches esperados.

        Returns:
            Cantidad total de batches que generará este sampler.
        """
        total = 0

        for indices in self.buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)

        return total