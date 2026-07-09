#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/tsp/config.py

"""Constantes del subpaquete ``optimization.classical.tsp``.

Este módulo centraliza parámetros por defecto usados por los solvers y
heurísticas TSP clásicas.
"""

# Tiempo máximo por defecto para OR-Tools.
TSP_DEFAULT_TIME_LIMIT = 30.0

# El TSP puro se modela como un problema de un único vehículo.
TSP_DEFAULT_VEHICLES = 1

# OR-Tools trabaja con costos enteros; este factor conserva más precisión.
TSP_DISTANCE_SCALING_FACTOR = 10000

# Ciudad inicial/depot usada por el modelo de ruteo.
TSP_DEPOT_INDEX = 0