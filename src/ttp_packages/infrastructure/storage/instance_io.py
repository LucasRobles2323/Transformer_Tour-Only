#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/instance_io.py

"""Carga de instancias TTP desde archivos ``.ttp``.

Este módulo parsea archivos de instancia del Traveling Thief Problem, convierte
índices base 1 a base 0, construye entidades de dominio y precalcula la matriz
de distancias de la instancia.
"""

from __future__ import annotations

from typing import List

from src.ttp_packages.domain.entities import City, Item
from src.ttp_packages.domain.instance import TTPInstance

from .paths import get_inst_file_path


def load_instance(fname: str) -> TTPInstance:
    """Carga una instancia TTP desde un archivo ``.ttp``.

    Lee los metadatos, ciudades e ítems de una instancia TTP almacenada en disco.
    Durante el parseo convierte los índices base 1 del archivo a índices base 0,
    vincula cada ítem con su ciudad y precalcula la matriz de distancias.

    Args:
        fname: Nombre del archivo de instancia, por ejemplo ``"a280_n279.ttp"``.

    Returns:
        Instancia TTP lista para ser usada por los módulos de dominio,
        evaluación u optimización.

    Raises:
        FileNotFoundError: Si el archivo no existe o está vacío.
        ValueError: Si la cantidad de ciudades o ítems leídos no coincide con
            los metadatos declarados, o si un ítem referencia una ciudad inválida.
    """
    path = get_inst_file_path(fname)   # Path directo

    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"El archivo {path} no existe o está vacío.")

    # ---- Metadatos ----
    n_cities = 0
    m_items = 0
    capacity = 0
    min_speed = 0.0
    max_speed = 1.0
    rent = 0.0

    # ---- Datos ----
    cities: List[City] = []
    items: List[Item] = []

    in_node_section = False
    in_items_section = False

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # 1) Header
            if line.startswith("DIMENSION"):
                n_cities = int(line.split()[-1])
                continue
            if line.startswith("NUMBER OF ITEMS"):
                m_items = int(line.split()[-1])
                continue
            if line.startswith("CAPACITY OF KNAPSACK"):
                capacity = int(line.split()[-1])
                continue
            if line.startswith("MIN SPEED"):
                min_speed = float(line.split()[-1])
                continue
            if line.startswith("MAX SPEED"):
                max_speed = float(line.split()[-1])
                continue
            if line.startswith("RENTING RATIO"):
                rent = float(line.split()[-1])
                continue

            # 2) Secciones
            if line.startswith("NODE_COORD_SECTION"):
                in_node_section = True
                in_items_section = False
                continue
            if line.startswith("ITEMS SECTION"):
                in_items_section = True
                in_node_section = False
                continue

            # 3) NODE_COORD_SECTION: "<index> <x> <y>"
            if in_node_section:
                parts = line.split()
                if len(parts) < 3:
                    continue

                idx_1b = int(float(parts[0]))     # por si viene con ".0"
                x = float(parts[1])
                y = float(parts[2])

                city_id = idx_1b - 1              # 1-based -> 0-based
                cities.append(City(city_id, x, y))

                if n_cities and len(cities) == n_cities:
                    in_node_section = False
                continue

            # 4) ITEMS SECTION: "<index> <profit> <weight> <node_index>"
            if in_items_section:
                parts = line.split()
                if len(parts) < 4:
                    continue

                idx_1b = int(float(parts[0]))
                profit = float(parts[1]) 
                weight = float(parts[2]) 
                assigned_node_1b = int(float(parts[3]))

                item_id = idx_1b - 1
                city_id = assigned_node_1b - 1

                new_item = Item(item_id, city_id, profit, weight)
                items.append(new_item)

                # Las instancias están normalizadas: city_id coincide con la posición en cities.
                if not 0 <= city_id < len(cities):
                    raise ValueError(
                        f"El ítem {item_id} referencia una ciudad inválida: {city_id}."
                    )
                
                # City.items almacena IDs de ítems asociados a la ciudad.
                cities[city_id].items.append(item_id)

                continue

    # ---- Validaciones de consistencia ----
    if n_cities and len(cities) != n_cities:
        raise ValueError(f"Se esperaban {n_cities} ciudades, pero se leyeron {len(cities)}.")

    if m_items and len(items) != m_items:
        raise ValueError(f"Se esperaban {m_items} items, pero se leyeron {len(items)}.")

    # Asegurar orden por id
    cities.sort(key=lambda c: c.id)
    items.sort(key=lambda it: it.id)

    instance = TTPInstance(
        path=path,
        name=fname,
        cities=cities,
        items=items,
        n_cities=n_cities if n_cities else len(cities),
        m_items=m_items if m_items else len(items),
        capacity=capacity,
        rent_per_time=rent,
        min_speed=min_speed,
        max_speed=max_speed,
        distance_matrix=None,
    )
    instance.create_distance_matrix()

    return instance