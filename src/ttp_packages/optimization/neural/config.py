#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/neural/config.py

"""Constantes compartidas del subpaquete ``optimization.neural``.

Este módulo centraliza valores usados durante inferencia y decodificación
neuronal de tours.
"""

# Modos de construcción de máscara compatibles con ml_data.transforms.augment.
MASK_MODE_DENSE = "dense"
MASK_MODE_KNN = "knn"

# Valor usado como penalización para bloquear aristas durante el decoding.
NEG_INF = -1e9