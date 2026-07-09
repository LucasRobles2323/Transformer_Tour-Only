#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/ttp/cs2sa_r/route_cache.py

"""Estructura de caché para evaluación incremental de rutas TTP.

Este módulo define ``RouteCache``, un contenedor liviano usado por CS2SA-R para
guardar el estado físico y objetivo de una solución TTP.

La caché permite que movimientos locales, como bit-flips de packing o 2-opt del
tour, puedan evaluarse de forma incremental sin recalcular toda la solución desde
cero en cada intento.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteCache:
    """Estado acumulado de una solución TTP.

    Attributes:
        tour: Secuencia de ciudades en orden de visita.
        packing: Lista binaria de ítems, donde ``1`` indica seleccionado y ``0``
            indica no seleccionado.
        t_acc: Tiempo acumulado después de recorrer el arco que sale de cada posición.
        w_acc: Peso acumulado al salir de cada posición del tour.
        t_reg: Tiempo local de cada arco ``tour[i] -> tour[(i + 1) % N]``.
        w_reg: Peso recogido en cada posición del tour.
        pos_in_tour: Mapa ``city_id -> posición`` para ubicar ciudades en O(1).
        g_profit: Profit total de los ítems seleccionados.
        g_weight: Peso total final de la mochila.
        f_time: Tiempo total del recorrido.
        G_gain: Valor objetivo TTP, calculado como
            ``profit - rent_per_time * time``.
    """

    # Solución discreta actual.
    tour: list[int]
    packing: list[int]

    # Series acumuladas y locales por posición del tour.
    t_acc: list[float]
    w_acc: list[float]
    t_reg: list[float]
    w_reg: list[float]

    # Índice inverso para ubicar rápidamente la posición de una ciudad en el tour.
    pos_in_tour: dict[int, int]

    # Métricas agregadas de la solución.
    g_profit: float
    g_weight: float
    f_time: float
    G_gain: float