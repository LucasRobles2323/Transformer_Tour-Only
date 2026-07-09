#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/modeling/config.py

"""Configuración y constantes compartidas del paquete ``modeling``.

Este módulo define los parámetros estructurales de la arquitectura TTP, valores
de estabilidad numérica y mensajes de error reutilizados por capas y modelos.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TTPArchitectureParams:
    """Parámetros estructurales de la arquitectura TTP.

    Attributes:
        d_model: Dimensión de los embeddings latentes. Debe ser par si se usa
            codificación posicional seno/coseno.
        n_heads: Número de cabezales de atención.
        n_layers: Número de capas Transformer del encoder.
        d_ff: Dimensión interna de la red feed-forward del Transformer.
        dropout: Probabilidad de dropout.
        sinkhorn_iter: Número de iteraciones de Sinkhorn.
        sink_tau: Temperatura usada por Sinkhorn.
        coupling_iters: Número de iteraciones de acoplamiento entre ramas.
        node_feature_set: Conjunto de features por ciudad. ``"basic"`` conserva
            el comportamiento anterior; ``"ttp_v1"`` agrega señales TTP-aware.
        edge_feature_mode: Modo de atributos de arista. ``"none"`` conserva el
            comportamiento anterior; ``"distance_v1"`` agrega señales simples
            derivadas de distancias, velocidades y renta.
    """

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.1
    sinkhorn_iter: int = 80
    sink_tau: float = 0.1
    coupling_iters: int = 1

    # Nuevos modos compatibles hacia atrás.
    node_feature_set: str = "basic"
    edge_feature_mode: str = "none"


# Constantes de estabilidad numérica.
DEFAULT_EPSILON = 1e-9
DEFAULT_MAX_LOG = 10.0
PE_DEFAULT_MAX_LEN = 5000
PE_DIV_TERM_BASE = 10000.0

# Mensajes de error de TTPModel.
ERR_MSG_MISSING_CORE_INPUTS = "coords, W y node_mask son obligatorios."
ERR_MSG_SINKHORN_MASK_MISSING = "sinkhorn_mask es obligatorio."

# Mensajes de error de Sinkhorn.
ERR_MSG_SINKHORN_SHAPE = "Los logits deben ser cuadrados (B,N,N). Se recibió {shape}"