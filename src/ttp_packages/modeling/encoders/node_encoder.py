#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/modeling/encoders/node_encoder.py

"""Encoder de nodos para representar ciudades TTP.

Este módulo transforma coordenadas de ciudades en embeddings enriquecidos usando
proyección lineal, codificación posicional, Transformer Encoder y fusión con el
embedding de capacidad.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .positional_encoding import PositionalEncoding


class NodeEncoder(nn.Module):
    """Encoder de ciudades para el branch de nodos.

    Args:
        d_in: Dimensión de entrada de cada ciudad. Normalmente ``2`` para
            coordenadas ``(x, y)``.
        d_model: Dimensión latente usada por el modelo.
        n_heads: Número de cabezales de atención del Transformer.
        n_layers: Número de capas del Transformer Encoder.
        d_ff: Dimensión interna de la red feed-forward del Transformer.
        dropout: Probabilidad de dropout.
    """

    def __init__(
        self,
        d_in: int = 2,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Proyección Inicial: Convierte coordenadas 2D a espacio latente d_model
        self.inp = nn.Linear(d_in, d_model)

        # 2. Codificación Posicional:
        # Como el Transformer no entiende orden ni secuencia por sí mismo, 
        # necesitamos inyectar una señal que indique "posición" (aunque en grafos esto es relativo).
        self.pe = PositionalEncoding(d_model, dropout=dropout)

        # 3. Bloque Transformer Encoder
        # batch_first=True es crucial para que coincida con la forma (Batch, N, D)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        # Desactiva NestedTensor interno: los batches ya son densos y homogéneos.
        self.enc = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # 4. Normalización final para estabilizar la salida antes de la siguiente etapa
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        coords: Tensor,
        cap_embed: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Codifica ciudades y fusiona la información de capacidad.

        Args:
            coords: Coordenadas de ciudades con shape ``(B, N, 2)``.
            cap_embed: Embedding de capacidad con shape ``(B, 1, D)``.
            mask: Máscara booleana opcional con shape ``(B, N)``. En
                ``src_key_padding_mask``, True indica posiciones que el
                Transformer debe ignorar.

        Returns:
            Embeddings de ciudades con capacidad fusionada, shape ``(B, N, D)``.
        """
        hidden = self.inp(coords)  # (B, N, D)
        hidden = self.pe(hidden)  # (B, N, D)

        if mask is not None:
            hidden = self.enc(hidden, src_key_padding_mask=mask)
        else:
            hidden = self.enc(hidden)

        hidden = self.norm(hidden)  # (B, N, D)

        # Broadcasting: cap_embed pasa de (B, 1, D) a (B, N, D).
        return hidden + cap_embed