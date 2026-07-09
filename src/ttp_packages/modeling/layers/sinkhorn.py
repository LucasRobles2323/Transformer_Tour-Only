#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/modeling/layers/sinkhorn.py

"""Capa Sinkhorn diferenciable para matrices de transición.

Este módulo transforma logits de aristas ``(B, N, N)`` en matrices aproximadas
bi-estocásticas, con soporte para máscaras binarias de aristas permitidas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.ttp_packages.modeling.config import (
    DEFAULT_EPSILON,
    ERR_MSG_SINKHORN_SHAPE,
)


class Sinkhorn(nn.Module):
    """Normalización Sinkhorn-Knopp diferenciable.

    Args:
        n_iter: Número de iteraciones alternando normalización por filas y columnas.
        tau: Temperatura aplicada a los logits antes de exponenciar.
        eps: Epsilon usado para estabilidad numérica.
    """

    def __init__(
        self,
        n_iter: int = 50,
        tau: float = 0.1,
        eps: float = DEFAULT_EPSILON,
    ):
        """Inicializa la capa Sinkhorn.

        Args:
            n_iter: Número de iteraciones de normalización fila/columna.
            tau: Temperatura. Valores menores producen distribuciones más nítidas.
            eps: Valor pequeño para evitar divisiones por cero.
        """
        super().__init__()

        self.n_iter = int(n_iter)
        self.tau = float(tau)
        self.eps = float(eps)

    def forward(self, logits: Tensor, mask: Tensor | None = None) -> Tensor:
        """Aplica normalización Sinkhorn a logits cuadrados.

        Args:
            logits: Tensor de logits con shape ``(B, N, N)``.
            mask: Máscara binaria opcional donde ``1`` indica arista permitida y
                ``0`` arista bloqueada. Puede tener shape ``(N, N)``,
                ``(1, N, N)`` o ``(B, N, N)``.

        Returns:
            Matriz aproximadamente bi-estocástica con shape ``(B, N, N)``.

        Raises:
            ValueError: Si ``logits`` no tiene shape cuadrado ``(B, N, N)``.
        """
        batch_size, n_rows, n_cols = logits.shape
        if n_rows != n_cols:
            raise ValueError(ERR_MSG_SINKHORN_SHAPE.format(shape=logits.shape))

        n_cities = n_rows
        scaled_logits = logits / max(self.tau, self.eps)

        allowed_mask: Tensor | None = None
        allowed_values: Tensor | None = None

        if mask is not None:
            # Expandimos mask a (B,N,N)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            elif mask.dim() == 3 and mask.size(0) != batch_size:
                mask = mask[:1].expand(batch_size, -1, -1)

            mask = mask[:, :n_cities, :n_cities].to(device=logits.device)
            allowed_mask = mask > 0.0
            allowed_values = allowed_mask.to(dtype=logits.dtype)

            # Enmascarado duro: 
            #      - bloqueamos con un valor muy negativo
            #      - las aristas bloqueadas no participan en exp().
            neg_inf = torch.finfo(scaled_logits.dtype).min
            scaled_logits = scaled_logits.masked_fill(~allowed_mask, neg_inf)

        # Restar el máximo estabiliza exp() y no cambia la normalización posterior.
        scaled_logits = scaled_logits - scaled_logits.amax(dim=(1, 2), keepdim=True)

        probs = torch.exp(scaled_logits)

        if allowed_values is not None:
            # Garantiza 0 exacto en aristas bloqueadas después de exponenciar.
            probs = probs * allowed_values

        for _ in range(self.n_iter):
            # Normalizar filas
            probs = probs / (probs.sum(dim=2, keepdim=True) + self.eps)
            if allowed_values is not None:
                probs = probs * allowed_values

            # Normalizar columnas
            probs = probs / (probs.sum(dim=1, keepdim=True) + self.eps)
            if allowed_values is not None:
                probs = probs * allowed_values

        return probs