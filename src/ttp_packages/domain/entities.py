#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/domain/entities.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from .constants import DECIMALS

@dataclass
class City:
    """Representa una ciudad del problema TTP.

    Attributes:
        id: Identificador único de la ciudad, normalmente en el rango ``0..N-1``.
        x: Coordenada horizontal de la ciudad.
        y: Coordenada vertical de la ciudad.
        items: Índices globales de los ítems disponibles en esta ciudad.
    """
    id: int
    x: float
    y: float
    items: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"City {self.id} (x={self.x:.{DECIMALS}f}, y={self.y:.{DECIMALS}f}) items:{self.items}"

@dataclass
class Item:
    """Representa un ítem que puede recogerse durante el tour.

    Attributes:
        id: Identificador global del ítem, normalmente en el rango ``0..M-1``.
        city_id: Identificador de la ciudad donde está disponible el ítem.
        profit: Ganancia obtenida al recoger el ítem.
        weight: Peso agregado a la mochila al recoger el ítem.
    """
    id: int
    city_id: int
    profit: float
    weight: float

    def __repr__(self) -> str:
        return (f"Item {self.id} [City: {self.city_id}] "
                f"(P={self.profit:.{DECIMALS}f}, W={self.weight:.{DECIMALS}f})")