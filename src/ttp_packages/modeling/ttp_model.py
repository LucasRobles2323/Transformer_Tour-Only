#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/modeling/ttp_model.py

"""Modelo principal TTP para predicción de tours.

Este módulo define ``TTPModel``, una arquitectura PyTorch que combina
coordenadas de ciudades, información agregada de ítems y variables globales del
problema para producir logits de aristas y probabilidades de transición.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .branches.tour_branch import TourBranch
from .config import (
    DEFAULT_EPSILON,
    DEFAULT_MAX_LOG,
    ERR_MSG_MISSING_CORE_INPUTS,
    ERR_MSG_SINKHORN_MASK_MISSING,
    TTPArchitectureParams,
)
from .encoders.node_encoder import NodeEncoder
from .utils.features import as_b1, log_feat01

class TTPModel(nn.Module):
    """Modelo tour-only consciente del contexto TTP.

    El modelo agrega señales de ítems por ciudad, codifica las ciudades con un
    Transformer y genera una matriz de transición mediante el branch de tour.

    Args:
        params: Hiperparámetros estructurales de la arquitectura.
    """

    def __init__(self, params: TTPArchitectureParams):
        """Inicializa los submódulos del modelo.

        Args:
            params: Parámetros de arquitectura.
        """
        super().__init__()

        # Guarda parámetros para que inferencia/checkpoints puedan consultar
        # cómo fue construido el modelo.
        self.params = params

        self.node_feature_set = str(getattr(params, "node_feature_set", "basic"))
        self.edge_feature_mode = str(getattr(params, "edge_feature_mode", "none"))

        self.node_feature_dim = self._node_feature_dim(self.node_feature_set)
        self.edge_attr_dim = self._edge_attr_dim(self.edge_feature_mode)

        # Red para procesar variables globales (W, R, Vmin, Vmax).
        self.global_embed = nn.Sequential(
            nn.Linear(4, params.d_model),
            nn.ReLU(),
            nn.Linear(params.d_model, params.d_model),
        )

        # El encoder recibe una dimensión dependiente de node_feature_set.
        self.encoder = NodeEncoder(
            d_in=self.node_feature_dim,
            d_model=params.d_model,
            n_heads=params.n_heads,
            n_layers=params.n_layers,
            d_ff=params.d_ff,
            dropout=params.dropout,
        )

        # El branch de tour puede recibir atributos de arista si edge_feature_mode
        # lo requiere.
        self.tour = TourBranch(
            d_model=params.d_model,
            sinkhorn_iter=params.sinkhorn_iter,
            sink_tau=params.sink_tau,
            edge_attr_dim=self.edge_attr_dim,
        )

    @staticmethod
    def _node_feature_dim(node_feature_set: str) -> int:
        """Devuelve la dimensión de entrada del encoder según el set de features.

        Args:
            node_feature_set: Nombre del conjunto de atributos de nodo.

        Returns:
            Dimensión de las features por ciudad.

        Raises:
            ValueError: Si el conjunto de features no está soportado.
        """
        if node_feature_set == "basic":
            # coords + profit_sum + weight_sum + count
            return 5

        if node_feature_set == "ttp_v1":
            # coords
            # + profit_sum + weight_sum + count
            # + profit_mean + weight_mean
            # + density_mean + density_max
            # + has_items
            # + profit_over_capacity + weight_over_capacity
            return 12

        raise ValueError(
            "node_feature_set inválido. Usa 'basic' o 'ttp_v1'. "
            f"Recibido: {node_feature_set}"
        )

    @staticmethod
    def _edge_attr_dim(edge_feature_mode: str) -> int:
        """Devuelve la dimensión de atributos de arista según el modo elegido.

        Args:
            edge_feature_mode: Nombre del modo de atributos de arista.

        Returns:
            Dimensión de atributos de arista.

        Raises:
            ValueError: Si el modo de atributos de arista no está soportado.
        """
        if edge_feature_mode == "none":
            return 0

        if edge_feature_mode == "distance_v1":
            return 4

        raise ValueError(
            "edge_feature_mode inválido. Usa 'none' o 'distance_v1'. "
            f"Recibido: {edge_feature_mode}"
        )

    def forward(
        self,
        coords: Tensor,
        W: Tensor,
        node_mask: Tensor,
        sinkhorn_mask: Tensor,
        item_city: Tensor,
        item_profit: Tensor,
        item_weight: Tensor,
        item_mask: Tensor,
        min_speed: Tensor | float,
        max_speed: Tensor | float,
        rent_per_time: Tensor | float,
        dist_matrix: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Ejecuta el paso forward del modelo.

        Args:
            coords: Coordenadas normalizadas de ciudades con shape ``(B, N, 2)``.
            W: Capacidad de la mochila con shape ``(B, 1)`` o compatible.
            node_mask: Máscara de ciudades válidas con shape ``(B, N)``.
            sinkhorn_mask: Máscara de aristas para Sinkhorn con shape compatible
                con ``(B, N, N)``.
            item_city: Ciudad asociada a cada ítem con shape ``(B, M)``.
            item_profit: Beneficio de cada ítem con shape ``(B, M)``.
            item_weight: Peso de cada ítem con shape ``(B, M)``.
            item_mask: Máscara de ítems válidos con shape ``(B, M)``.
            min_speed: Velocidad mínima, escalar o tensor compatible con
                shape ``(B, 1)``.
            max_speed: Velocidad máxima, escalar o tensor compatible con
                shape ``(B, 1)``.
            rent_per_time: Renta por unidad de tiempo, escalar o tensor
                compatible con shape ``(B, 1)``.
            dist_matrix: Matriz opcional de distancias con shape ``(B, N, N)``.
                Es obligatoria si ``edge_feature_mode == "distance_v1"``.

        Returns:
            Tupla ``(transition_probs, edge_logits)``:
                - ``transition_probs``: Matriz de transición con shape
                  ``(B, N, N)``.
                - ``edge_logits``: Logits de aristas con shape ``(B, N, N)``.

        Raises:
            ValueError: Si faltan entradas requeridas o si ``distance_v1`` se
                usa sin ``dist_matrix``.
        """
        if coords is None or W is None or node_mask is None:
            raise ValueError(ERR_MSG_MISSING_CORE_INPUTS)

        if sinkhorn_mask is None:
            raise ValueError(ERR_MSG_SINKHORN_MASK_MISSING)

        device = coords.device
        batch_size, n_cities, _ = coords.shape
        eps = DEFAULT_EPSILON

        # ------------------------------------------------------------------
        # 1) Agregación de señales de ítems por ciudad.
        # ------------------------------------------------------------------
        item_valid_mask = item_mask.float().to(device)

        item_city_clamped = item_city.to(
            device=device,
            dtype=torch.long,
        ).clamp(
            min=0,
            max=n_cities - 1,
        )

        item_profit = item_profit.float().to(device)
        item_weight = item_weight.float().to(device)

        city_profit_raw = torch.zeros(batch_size, n_cities, device=device)
        city_weight_raw = torch.zeros(batch_size, n_cities, device=device)
        city_count = torch.zeros(batch_size, n_cities, device=device)

        city_profit_raw.scatter_add_(
            1,
            item_city_clamped,
            item_profit * item_valid_mask,
        )
        city_weight_raw.scatter_add_(
            1,
            item_city_clamped,
            item_weight * item_valid_mask,
        )
        city_count.scatter_add_(
            1,
            item_city_clamped,
            item_valid_mask,
        )

        # Capacidad con shape (B, 1) para normalizaciones seguras.
        capacity_b1 = W.float().to(device).view(batch_size, -1)[:, :1].clamp_min(eps)

        # ------------------------------------------------------------------
        # 2) Construcción de features de nodo.
        # ------------------------------------------------------------------
        if self.node_feature_set == "basic":
            # Comportamiento anterior:
            # coords + profit total + weight total + count.
            node_features = torch.cat(
                [
                    coords,
                    torch.log1p(city_profit_raw).unsqueeze(-1),
                    torch.log1p(city_weight_raw).unsqueeze(-1),
                    city_count.unsqueeze(-1),
                ],
                dim=-1,
            )

        elif self.node_feature_set == "ttp_v1":
            city_count_safe = city_count.clamp_min(1.0)

            profit_mean = city_profit_raw / city_count_safe
            weight_mean = city_weight_raw / city_count_safe

            # Densidad profit/peso por ítem. Los ítems inválidos quedan anulados.
            item_density = item_profit / item_weight.clamp_min(eps)
            item_density = item_density * item_valid_mask

            density_sum = torch.zeros(batch_size, n_cities, device=device)
            density_sum.scatter_add_(
                1,
                item_city_clamped,
                item_density,
            )

            density_max = torch.zeros(batch_size, n_cities, device=device)
            density_max.scatter_reduce_(
                1,
                item_city_clamped,
                item_density,
                reduce="amax",
                include_self=True,
            )

            density_mean = density_sum / city_count_safe
            has_items = (city_count > 0).float()

            node_features = torch.cat(
                [
                    coords,
                    torch.log1p(city_profit_raw).unsqueeze(-1),
                    torch.log1p(city_weight_raw).unsqueeze(-1),
                    city_count.unsqueeze(-1),
                    torch.log1p(profit_mean).unsqueeze(-1),
                    torch.log1p(weight_mean).unsqueeze(-1),
                    torch.log1p(density_mean).unsqueeze(-1),
                    torch.log1p(density_max).unsqueeze(-1),
                    has_items.unsqueeze(-1),
                    (city_profit_raw / capacity_b1).unsqueeze(-1),
                    (city_weight_raw / capacity_b1).unsqueeze(-1),
                ],
                dim=-1,
            )

        else:
            raise ValueError(
                "node_feature_set inválido. Usa 'basic' o 'ttp_v1'. "
                f"Recibido: {self.node_feature_set}"
            )

        if node_features.size(-1) != self.node_feature_dim:
            raise RuntimeError(
                "La dimensión de node_features no coincide con node_feature_dim. "
                f"Recibido {node_features.size(-1)}, esperado {self.node_feature_dim}."
            )

        # ------------------------------------------------------------------
        # 3) Construcción opcional de atributos de arista.
        # ------------------------------------------------------------------
        edge_attr = None

        if self.edge_feature_mode == "distance_v1":
            if dist_matrix is None:
                raise ValueError(
                    "edge_feature_mode='distance_v1' requiere dist_matrix. "
                    "Activa compute_dist_matrix=True en augment_batch_on_device/training."
                )

            dist_matrix = dist_matrix.float().to(device)

            if dist_matrix.dim() != 3 or dist_matrix.shape != (
                batch_size,
                n_cities,
                n_cities,
            ):
                raise ValueError(
                    "dist_matrix debe tener shape (B,N,N). "
                    f"Recibido {tuple(dist_matrix.shape)}, "
                    f"esperado {(batch_size, n_cities, n_cities)}."
                )

            dist_max = dist_matrix.amax(dim=(1, 2), keepdim=True).clamp_min(eps)
            distance_norm = dist_matrix / dist_max

            min_speed_b1_for_edges = as_b1(min_speed, batch_size, device).clamp_min(eps)
            max_speed_b1_for_edges = as_b1(max_speed, batch_size, device).clamp_min(eps)
            rent_b1_for_edges = as_b1(rent_per_time, batch_size, device)

            edge_attr = torch.stack(
                [
                    distance_norm,
                    dist_matrix / max_speed_b1_for_edges.view(batch_size, 1, 1),
                    dist_matrix / min_speed_b1_for_edges.view(batch_size, 1, 1),
                    rent_b1_for_edges.view(batch_size, 1, 1) * distance_norm,
                ],
                dim=-1,
            )

        elif self.edge_feature_mode == "none":
            edge_attr = None

        else:
            raise ValueError(
                "edge_feature_mode inválido. Usa 'none' o 'distance_v1'. "
                f"Recibido: {self.edge_feature_mode}"
            )

        # ------------------------------------------------------------------
        # 4) Variables globales TTP.
        # ------------------------------------------------------------------
        min_speed_b1 = as_b1(min_speed, batch_size, device)
        max_speed_b1 = as_b1(max_speed, batch_size, device)
        rent_per_time_b1 = as_b1(rent_per_time, batch_size, device)
        capacity_log = log_feat01(
            W.float().to(device),
            max_log=DEFAULT_MAX_LOG,
        )

        global_features = torch.cat(
            [
                capacity_log,
                rent_per_time_b1,
                min_speed_b1,
                max_speed_b1,
            ],
            dim=-1,
        )  # (B, 4)

        global_embedding = self.global_embed(global_features).unsqueeze(1)

        # ------------------------------------------------------------------
        # 5) Encoder de ciudades.
        # ------------------------------------------------------------------
        node_mask = node_mask.to(device)
        padding_mask = ~node_mask.bool()

        hidden = self.encoder(
            node_features,
            global_embedding,
            mask=padding_mask,
        )

        # ------------------------------------------------------------------
        # 6) Branch de tour.
        # ------------------------------------------------------------------
        transition_probs, edge_logits = self.tour(
            h=hidden,
            node_mask=node_mask,
            sinkhorn_mask=sinkhorn_mask,
            edge_attr=edge_attr,
            train=self.training,
            eps=eps,
        )

        return transition_probs, edge_logits