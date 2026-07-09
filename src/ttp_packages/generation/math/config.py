#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/generation/math/config.py

"""Constantes del subpaquete generation.math.

Este módulo agrupa parámetros usados por los proxies matemáticos para estimar
el factor de renta ``R``: heurísticas TSP, aproximaciones de mochila y mensajes
de error numérico.
"""


# ---------------------------------------------------------------------------
# Parámetros de algoritmos
# ---------------------------------------------------------------------------

# Factor de error para FPTAS de mochila. Un valor menor mejora precisión, pero
# aumenta el tiempo y memoria requeridos.
DEFAULT_EPSILON = 0.05

# Límite de iteraciones para la mejora local 2-opt del tour proxy.
DEFAULT_2OPT_ITERS = 200

# Nodo inicial por defecto. En TTP suele ser el depósito.
DEFAULT_START_NODE = 0


# ---------------------------------------------------------------------------
# Umbrales para selección automática de mochila
# ---------------------------------------------------------------------------

KNAPSACK_THRESHOLD_ITEMS = 2000
KNAPSACK_THRESHOLD_CAPACITY = 200000


# ---------------------------------------------------------------------------
# Mensajes de error
# ---------------------------------------------------------------------------

ERR_CRITICAL_TIME = (
    "Error crítico: tiempo total <= 0. Revisar configuración de velocidades."
)