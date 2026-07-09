#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/modeling/heads/edge_heatmap_head.py

"""Cabezal de logits para aristas del tour.

Este módulo transforma embeddings de ciudades en una matriz densa de logits
``(B, N, N)``, donde cada valor representa la afinidad dirigida de viajar desde
una ciudad origen hacia una ciudad destino.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EdgeHeatmapHead(nn.Module):
    """Cabezal de mapa de calor para aristas dirigidas.

    Args:
        d_model: Dimensión de los embeddings de entrada.
        edge_attr_dim: Dimensión de atributos opcionales de arista. Si es ``0``,
            se conserva el comportamiento anterior.
    """

    def __init__(
        self,
        d_model: int = 128,
        edge_attr_dim: int = 0,
    ):
        """Inicializa las proyecciones del cabezal de aristas.

        Args:
            d_model: Dimensión de los embeddings de ciudad.
            edge_attr_dim: Dimensión de atributos de arista. Si es ``0``, no
                se usan atributos externos.
        """
        super().__init__()

        self.d_model = int(d_model)
        self.edge_attr_dim = int(edge_attr_dim)

        # Proyección para el nodo origen.
        self.q = nn.Linear(d_model, d_model, bias=False)

        # Proyección para el nodo destino.
        self.k = nn.Linear(d_model, d_model, bias=False)

        # Proyección opcional de atributos de arista al espacio d_model.
        self.edge_proj: nn.Linear | None = None
        if self.edge_attr_dim > 0:
            self.edge_proj = nn.Linear(self.edge_attr_dim, d_model, bias=False)

        # Proyección final a score escalar.
        self.out = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        h: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Calcula logits dirigidos para cada par de ciudades.

        Args:
            h: Embeddings de ciudades con shape ``(B, N, D)``.
            edge_attr: Atributos opcionales de arista con shape ``(B, N, N, E)``.
                Solo se aceptan si ``edge_attr_dim > 0``.

        Returns:
            Tensor de logits con shape ``(B, N, N)``.

        Raises:
            ValueError: Si la configuración de atributos de arista no coincide
                con el tensor recibido.
            RuntimeError: Si la salida no tiene shape ``(B, N, N)``.
        """
        if h.ndim != 3:
            raise ValueError(
                "EdgeHeatmapHead esperaba h con shape (B,N,D). "
                f"Recibido {tuple(h.shape)}."
            )

        batch_size, n_cities, _ = h.shape

        query = self.q(h)  # (B, N, D)
        key = self.k(h)    # (B, N, D)

        pair_features = query.unsqueeze(2) + key.unsqueeze(1)  # (B, N, N, D)

        if self.edge_attr_dim > 0:
            if edge_attr is None:
                raise ValueError(
                    "edge_attr es obligatorio cuando edge_attr_dim > 0."
                )

            if edge_attr.ndim != 4:
                raise ValueError(
                    "edge_attr debe tener shape (B,N,N,E). "
                    f"Recibido {tuple(edge_attr.shape)}."
                )

            if edge_attr.shape[:3] != (batch_size, n_cities, n_cities):
                raise ValueError(
                    "edge_attr debe coincidir con batch y ciudades de h. "
                    f"edge_attr={tuple(edge_attr.shape)}, "
                    f"h={tuple(h.shape)}."
                )

            if edge_attr.size(-1) != self.edge_attr_dim:
                raise ValueError(
                    "La última dimensión de edge_attr no coincide con edge_attr_dim. "
                    f"Recibido E={edge_attr.size(-1)}, esperado E={self.edge_attr_dim}."
                )

            if self.edge_proj is None:
                raise RuntimeError(
                    "edge_proj no fue inicializado aunque edge_attr_dim > 0."
                )

            pair_features = pair_features + self.edge_proj(edge_attr)

        elif edge_attr is not None:
            raise ValueError(
                "edge_attr fue entregado, pero edge_attr_dim=0. "
                "Usa edge_feature_mode='distance_v1' o no entregues edge_attr."
            )

        scores = self.out(torch.tanh(pair_features))  # (B, N, N, 1)
        logits = scores.squeeze(-1)  # (B, N, N)

        if logits.dim() != 3:
            raise RuntimeError(
                f"EdgeHeatmapHead esperaba logits 3D, recibió shape={tuple(logits.shape)}."
            )

        return logits