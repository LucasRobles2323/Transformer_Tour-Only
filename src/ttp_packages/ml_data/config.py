#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/config.py

"""Constantes compartidas del paquete ``ml_data``.

Define claves de payloads, modos de máscara y valores por defecto usados para
representación, datasets y entrenamiento.
"""

# Modos de Máscara para Sinkhorn/Atención
MASK_MODE_DENSE = "dense"
MASK_MODE_KNN = "knn"

# Configuración por defecto de Aumentación
DEFAULT_KNN_K = 32
DEFAULT_EPS = 1e-9

# Tipos de Datos
DTYPE_FLOAT = "float32"
DTYPE_LONG = "long"

# Nombres de Claves de entrada (Inputs)
KEY_COORDS_RAW = "coords_raw"
KEY_COORDS_NORM = "coords"
KEY_CAPACITY = "W"
KEY_ITEM_CITY = "item_city"
KEY_ITEM_PROFIT = "item_profit"
KEY_ITEM_WEIGHT = "item_weight"
KEY_MIN_SPEED = "min_speed"
KEY_MAX_SPEED = "max_speed"
KEY_RENT = "rent_per_time"
# Nombres de Claves de entrada derivadas durante augmentación
KEY_SINKHORN_MASK = "sinkhorn_mask"  # Máscara usada por el modelo/Sinkhorn
KEY_LOSS_MASK = "loss_mask"          # Máscara usada por la pérdida supervisada
KEY_DECODER_MASK = "decoder_mask"    # Máscara usada por decoding/inferencia
KEY_DIST_MATRIX = "dist_matrix"

# Nombres de Claves de salida (Teacher)
KEY_TOUR_NEXT = "tour_next"
KEY_TOUR_ADJ = "tour_adj"
KEY_PICKS = "picks"
KEY_PROFIT = "profit"
KEY_TIME = "time"
KEY_OBJECTIVE = "objective"