#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/packing/config.py

"""Constantes del subpaquete ``optimization.classical.packing``.

Este módulo centraliza los valores por defecto usados para optimizar el packing
cuando el tour ya está fijo.
"""

# Presupuesto total usado por defecto para mejorar packing sobre un tour fijo.
FIXED_TOUR_PACKING_TIME_S = 30.0

# Cantidad de reinicios independientes para buscar mejores packings.
FIXED_TOUR_PACKING_RESTARTS = 3

# Ruido pequeño usado para desempatar candidatos durante la construcción inicial.
FIXED_TOUR_PACKING_JITTER = 1e-9