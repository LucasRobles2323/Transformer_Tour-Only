#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/domain/instance.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from src.ttp_packages.infrastructure.logging import setup_logger

from .entities import City, Item
from .constants import DEBUG_LINE_WIDTH

# Inicialización del logger para este módulo específico
logger = setup_logger(__name__)

@dataclass
class TTPInstance:
    """Representa una instancia completa del Traveling Thief Problem.

    Attributes:
        path: Ruta o identificador del archivo fuente.
        name: Nombre de la instancia.
        cities: Lista de ciudades.
        items: Lista de ítems.
        n_cities: Cantidad declarada de ciudades.
        m_items: Cantidad declarada de ítems.
        capacity: Capacidad máxima de la mochila.
        rent_per_time: Factor de renta por unidad de tiempo.
        min_speed: Velocidad mínima.
        max_speed: Velocidad máxima.
        distance_matrix: Matriz de distancias ``N x N`` como ``np.ndarray``.
    """
    path: Path
    name: str
    cities: List[City] = field(default_factory=list)
    items: List[Item] = field(default_factory=list)
    n_cities: int = 0
    m_items: int = 0
    capacity: int = 0
    rent_per_time: float = 0.0
    min_speed: float = 0.0
    max_speed: float = 0.0
    distance_matrix: Optional[NDArray[np.float64]] = None

    def create_distance_matrix(self) -> None:
        """Calcula la matriz completa de distancias entre ciudades.

        La distancia TTP se calcula como la distancia euclidiana entre ciudades,
        redondeada hacia arriba con ``ceil``. La diagonal queda en cero porque la
        distancia de una ciudad a sí misma no se usa en el tour.

        Raises:
            ValueError: Si ``n_cities`` no coincide con la cantidad real de ciudades.
        """
        n = int(self.n_cities)

        if n != len(self.cities):
            raise ValueError(
                f"n_cities={n} no coincide con len(cities)={len(self.cities)}."
            )

        if n < 2:
            self.distance_matrix = np.zeros((n, n), dtype=np.float64)
            return

        coords = np.asarray([(city.x, city.y) for city in self.cities], dtype=np.float64)

        # Broadcasting:
        # coords[:, None, :] crea una vista (n, 1, 2)
        # coords[None, :, :] crea una vista (1, n, 2)
        # La resta produce todas las diferencias par-a-par entre ciudades.
        diffs = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(diffs * diffs, axis=2))

        self.distance_matrix = np.ceil(distances).astype(np.float64)
        logger.debug("Matriz de distancias de tamaño %sx%s calculada.", n, n)

    def print_summary(self):
        """Registra un resumen de integridad de la instancia.

        El reporte compara los metadatos declarados en la instancia con la cantidad
        real de ciudades e ítems cargados. Esto ayuda a detectar errores de parsing
        antes de evaluar soluciones.
        """
        real_n, real_m = len(self.cities), len(self.items)
        status_cities = "✅ OK" if self.n_cities == real_n else f"❌ ERR ({real_n})"
        status_items = "✅ OK" if self.m_items == real_m else f"❌ ERR ({real_m})"

        summary = [
            "\n" + "=" * DEBUG_LINE_WIDTH,
            f"RESUMEN TTP: {self.path.name}",
            "-" * DEBUG_LINE_WIDTH,
            f"{'Ciudades':<20} | {self.n_cities:<10} {status_cities}",
            f"{'Items':<20} | {self.m_items:<10} {status_items}",
            f"{'Capacidad':<20} | {self.capacity}",
            f"{'Renta (R)':<20} | {self.rent_per_time:.4f}",
            f"{'Velocidades':<20} | Min: {self.min_speed:.2f} / Max: {self.max_speed:.2f}",
            "=" * DEBUG_LINE_WIDTH
        ]
        logger.info("\n".join(summary))