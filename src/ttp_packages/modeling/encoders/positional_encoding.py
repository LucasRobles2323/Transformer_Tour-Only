#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/modeling/encoders/positional_encoding.py

"""Codificación posicional sinusoidal para embeddings de ciudades.

Este módulo agrega señales seno/coseno a los embeddings para que el Transformer
pueda distinguir posiciones dentro de la secuencia de nodos.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.ttp_packages.modeling.config import PE_DEFAULT_MAX_LEN, PE_DIV_TERM_BASE


class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal.

    Args:
        d_model: Dimensión del embedding.
        dropout: Probabilidad de dropout aplicada después de sumar la posición.
        max_len: Longitud máxima precomputada para la codificación.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = PE_DEFAULT_MAX_LEN,
    ):
        super().__init__()

        # Capa de dropout para regularización tras sumar la posición
        self.dropout = nn.Dropout(p=dropout)

        # 1. Generar vector de posiciones [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # 2. Calcular el término divisor (frecuencias decrecientes en escala logarítmica)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(PE_DIV_TERM_BASE) / d_model)
        )

        # 3. Crear matriz de posiciones vacía (max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        # 4. Asignar Seno a índices pares y Coseno a índices impares
        # Esto crea un patrón ondulatorio único para cada posición
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 5. Registrar como buffer
        # register_buffer le dice a PyTorch: "Esto es parte del estado del modelo (se guarda/carga),
        # pero NO es un parámetro aprendible (no tiene gradientes)".
        # Añadimos dimensión de batch: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Suma codificación posicional a los embeddings.

        Args:
            x: Embeddings de entrada con shape ``(B, N, D)``.

        Returns:
            Embeddings con información posicional y dropout aplicado.
        """
        # Cortamos la matriz 'pe' pre-calculada hasta la longitud actual de la secuencia (x.size(1))
        # x: (B, N, D)
        # pe: (1, N, D) -> Se transmite (broadcast) automáticamente a todo el batch
        x = x + self.pe[:, : x.size(1)]

        # Aplicamos dropout (importante para evitar depender demasiado de la posición exacta)
        return self.dropout(x)