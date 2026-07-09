#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/modeling/branches/tour_branch.py

"""Branch de tour para estimar probabilidades de transición.

Este módulo transforma embeddings de ciudades en logits de aristas y luego usa
Sinkhorn para producir una matriz de transición aproximadamente estocástica.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.ttp_packages.modeling.config import DEFAULT_EPSILON
from src.ttp_packages.modeling.heads.edge_heatmap_head import EdgeHeatmapHead
from src.ttp_packages.modeling.layers.sinkhorn import Sinkhorn
from src.ttp_packages.modeling.utils.masks import (
    allowed_nodes_from_node_mask,
    expand_bnn_mask,
)


class TourBranch(nn.Module):
    """Rama encargada de generar probabilidades de transición del tour.

    Args:
        d_model: Dimensión de los embeddings de ciudad.
        sinkhorn_iter: Número de iteraciones del algoritmo Sinkhorn.
        sink_tau: Temperatura usada por Sinkhorn.
        edge_attr_dim: Dimensión de atributos opcionales de arista. Si es ``0``,
            se conserva el comportamiento anterior.
    """

    def __init__(
        self,
        d_model: int = 128,
        sinkhorn_iter: int = 50,
        sink_tau: float = 0.1,
        edge_attr_dim: int = 0,
    ):
        super().__init__()

        self.edge_head = EdgeHeatmapHead(
            d_model=d_model,
            edge_attr_dim=edge_attr_dim,
        )
        self.sinkhorn = Sinkhorn(n_iter=sinkhorn_iter, tau=sink_tau)

    def forward(
        self,
        h: Tensor,
        node_mask: Tensor,
        sinkhorn_mask: Tensor,
        edge_attr: Tensor | None = None,
        train: bool = False,
        eps: float = DEFAULT_EPSILON,
    ) -> tuple[Tensor, Tensor]:
        """Calcula la matriz de transición del tour.

        Args:
            h: Embeddings de nodos con shape ``(B, N, D)``.
            node_mask: Máscara de nodos válidos con shape ``(B, N)``.
            sinkhorn_mask: Máscara para Sinkhorn con shape compatible con
                ``(B, N, N)``.
            edge_attr: Atributos opcionales de arista con shape ``(B, N, N, E)``.
            train: Si es True, conserva gradientes. Si es False, ejecuta el
                cálculo sin gradientes y retorna tensores desacoplados.
            eps: Valor pequeño para estabilidad numérica.

        Returns:
            Tupla ``(transition_probs, edge_logits)``:
                - ``transition_probs``: Matriz de transición row-stochastic con
                  shape ``(B, N, N)``.
                - ``edge_logits``: Logits de aristas antes de Sinkhorn con shape
                  ``(B, N, N)``.
        """
        device = h.device
        batch_size, n_cities, _ = h.shape

        sinkhorn_mask_expanded = expand_bnn_mask(
            sinkhorn_mask,
            B=batch_size,
            N=n_cities,
            device=device,
        )
        allowed_transitions = allowed_nodes_from_node_mask(node_mask)
        sinkhorn_mask_expanded = sinkhorn_mask_expanded * allowed_transitions

        # Mantiene compatibilidad con el parámetro público train, pero evita
        # confundirlo con self.training.
        grad_enabled = bool(train)

        with torch.set_grad_enabled(grad_enabled):
            edge_logits = self.edge_head(
                h,
                edge_attr=edge_attr,
            )
            transition_probs = self.sinkhorn(
                edge_logits,
                sinkhorn_mask_expanded,
            )

        if not grad_enabled:
            edge_logits = edge_logits.detach()
            transition_probs = transition_probs.detach()

        # Re-normaliza solo sobre transiciones válidas después de Sinkhorn.
        transition_probs = transition_probs * allowed_transitions

        # Agrega una pequeña masa diagonal para evitar filas completamente nulas.
        identity = torch.eye(n_cities, device=device).unsqueeze(0)
        transition_probs = transition_probs + (identity * allowed_transitions) * eps
        transition_probs = transition_probs / (
            transition_probs.sum(dim=-1, keepdim=True) + eps
        )

        return transition_probs, edge_logits