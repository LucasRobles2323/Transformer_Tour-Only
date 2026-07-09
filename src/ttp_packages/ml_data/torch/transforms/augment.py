#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/torch/transforms/augment.py

from __future__ import annotations

import torch
from typing import Any, Dict

from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.ml_data.config import (
    MASK_MODE_DENSE, 
    MASK_MODE_KNN, 
    DEFAULT_KNN_K, 
    DEFAULT_EPS,
    KEY_COORDS_RAW, 
    KEY_COORDS_NORM, 
    KEY_ITEM_CITY, 
    KEY_DIST_MATRIX,
    KEY_SINKHORN_MASK, 
    KEY_LOSS_MASK,
    KEY_DECODER_MASK,
    KEY_TOUR_NEXT,
    KEY_TOUR_ADJ
)

from .coords_norm import normalize_coords_minmax_batched



logger = setup_logger(__name__)


def build_tour_adj_from_next(tour_next: torch.Tensor) -> torch.Tensor:
    """Convierte un tensor de siguiente ciudad en una matriz de adyacencia dirigida.

    Args:
        tour_next (torch.Tensor): Tensor de forma (B, N) y tipo long con índices 
            de la siguiente ciudad a visitar.

    Returns:
        torch.Tensor: Matriz de adyacencia de forma (B, N, N) y tipo float32 
            con valor 1.0 en la posición (i, next[i]).

    Raises:
        ValueError: Si el tensor de entrada no tiene exactamente 2 dimensiones.
    """
    if tour_next.ndim != 2:
        error_msg = f"tour_next debe ser (B,N). Recibido {tuple(tour_next.shape)}."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    B, N = tour_next.shape
    A = torch.zeros((B, N, N), dtype=torch.float32, device=tour_next.device)
    A.scatter_(2, tour_next.unsqueeze(-1), 1.0)
    return A


def build_dense_mask(B: int, N: int, device: torch.device, allow_self: bool = False) -> torch.Tensor:
    """Construye una máscara densa (totalmente conectada) para atención/Sinkhorn.

    Args:
        B (int): Tamaño del batch.
        N (int): Número de nodos (ciudades).
        device (torch.device): Dispositivo donde se alojará el tensor (CPU/GPU).
        allow_self (bool, optional): Indica si se permiten conexiones de un nodo 
            hacia sí mismo (diagonal principal). Por defecto es False.

    Returns:
        torch.Tensor: Tensor de máscara de forma (B, N, N) y tipo float32.
    """
    mask = torch.ones((B, N, N), dtype=torch.float32, device=device)
    if not allow_self:
        idx = torch.arange(N, device=device)
        mask[:, idx, idx] = 0.0
    return mask


def build_knn_mask(coords: torch.Tensor, k: int, allow_self: bool, sym: bool) -> torch.Tensor:
    """Construye una máscara dispersa basada en los K vecinos más cercanos (KNN).

    Args:
        coords (torch.Tensor): Tensor de coordenadas espaciales de forma (B, N, 2).
        k (int): Número de vecinos más cercanos a considerar.
        allow_self (bool): Si es True, permite auto-conexiones en la máscara.
        sym (bool): Si es True, fuerza a que la máscara resultante sea simétrica.

    Returns:
        torch.Tensor: Máscara KNN de forma (B, N, N) y tipo float32.

    Raises:
        ValueError: Si las coordenadas no tienen la forma (B, N, 2).
    """
    if coords.ndim != 3 or coords.shape[-1] != 2:
        error_msg = f"coords debe ser (B,N,2). Recibido {tuple(coords.shape)}."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    B, N, _ = coords.shape
    k = int(max(1, min(int(k), N - 1)))

    d = torch.cdist(coords, coords, p=2)
    if not allow_self:
        idx = torch.arange(N, device=coords.device)
        d[:, idx, idx] = float("inf")

    nn_idx = torch.topk(d, k=k, dim=2, largest=False).indices  # (B,N,k)

    mask = torch.zeros((B, N, N), dtype=torch.float32, device=coords.device)
    ones = torch.ones((B, N, k), dtype=torch.float32, device=coords.device)
    mask.scatter_(2, nn_idx, ones)

    if sym:
        mask = torch.maximum(mask, mask.transpose(1, 2))

    if not allow_self:
        idx = torch.arange(N, device=coords.device)
        mask[:, idx, idx] = 0.0

    return mask


def compute_dist_matrix_from_raw(coords_raw: torch.Tensor) -> torch.Tensor:
    """Calcula la matriz de distancias euclidianas redondeadas hacia arriba.

    Args:
        coords_raw (torch.Tensor): Coordenadas originales sin normalizar (B, N, 2).

    Returns:
        torch.Tensor: Matriz de distancias de forma (B, N, N) y tipo float32,
            calculada como el techo (ceil) de la distancia euclidiana.
    """
    d = torch.cdist(coords_raw, coords_raw, p=2)
    return torch.ceil(d).to(torch.float32)


def augment_batch_on_device(
    batch: Dict[str, Any],
    mask_mode: str = MASK_MODE_DENSE,
    knn_k: int = DEFAULT_KNN_K,
    allow_self: bool = False,
    sym: bool = True,
    compute_dist_matrix: bool = True,
    compute_tour_adj: bool = True,
    eps: float = DEFAULT_EPS,
) -> Dict[str, Any]:
    """Aumenta un batch directamente en GPU reconstruyendo matrices O(N^2).

    Espera que el batch ya esté transferido al dispositivo. Agrega claves
    reconstruidas al vuelo para mantener el peso en disco reducido.

    La función distingue tres máscaras:

    - ``sinkhorn_mask``: máscara usada por el modelo/Sinkhorn. Debe ser
      reproducible en inferencia y no incorpora aristas teacher.
    - ``decoder_mask``: máscara usada por decodificación. Inicialmente coincide
      con ``sinkhorn_mask``.
    - ``loss_mask``: máscara usada por la pérdida supervisada. Puede incorporar
      aristas teacher para evitar targets bloqueados bajo KNN.

    Args:
        batch (Dict[str, Any]): Diccionario con los datos del batch.
        mask_mode (str, optional): Modo de construcción de máscara ('dense' o 'knn'). 
            Por defecto utiliza MASK_MODE_DENSE.
        knn_k (int, optional): Vecinos para máscara KNN. Por defecto DEFAULT_KNN_K.
        allow_self (bool, optional): Permite conexiones al mismo nodo. Por defecto False.
        sym (bool, optional): Fuerza simetría en máscaras KNN. Por defecto True.
        compute_dist_matrix (bool, optional): Indica si se debe generar la matriz 
            de distancias. Por defecto True.
        compute_tour_adj (bool, optional): Indica si se debe generar la matriz de 
            adyacencia del tour óptimo. Por defecto True.
        eps (float, optional): Valor epsilon para estabilidad numérica. 
            Por defecto DEFAULT_EPS.

    Returns:
        Dict[str, Any]: Batch mutado con coordenadas normalizadas, máscaras y matrices derivadas.

    Raises:
        ValueError: Si el modo de máscara especificado no es válido.
    """
    coords_raw = batch["inputs"][KEY_COORDS_RAW]
    device = coords_raw.device
    B, N, _ = coords_raw.shape
    M = batch["inputs"][KEY_ITEM_CITY].shape[1]

    coords = normalize_coords_minmax_batched(coords_raw, eps=eps)
    node_mask = torch.ones((B, N), dtype=torch.float32, device=device)
    item_mask = torch.ones((B, M), dtype=torch.float32, device=device)

    if compute_dist_matrix:
        dist_matrix = compute_dist_matrix_from_raw(coords_raw)
        batch["inputs"][KEY_DIST_MATRIX] = dist_matrix

    if mask_mode == MASK_MODE_DENSE:
        sinkhorn_mask = build_dense_mask(B=B, N=N, device=device, allow_self=allow_self)

    elif mask_mode == MASK_MODE_KNN:
        sinkhorn_mask = build_knn_mask(
            coords=coords,
            k=knn_k,
            allow_self=allow_self,
            sym=sym,
        )

    else:
        error_msg = f"mask_mode inválido: {mask_mode}. Usa '{MASK_MODE_DENSE}' o '{MASK_MODE_KNN}'."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # ------------------------------------------------------------------
    # Máscaras derivadas
    # ------------------------------------------------------------------
    # sinkhorn_mask:
    #   Máscara usada por el modelo y por Sinkhorn. Debe ser reproducible
    #   durante inferencia, por lo que NO incorpora aristas teacher.
    #
    # decoder_mask:
    #   Máscara usada posteriormente por el decoder. Por ahora coincide con
    #   la máscara base del modelo.
    #
    # loss_mask:
    #   Máscara usada únicamente por la pérdida supervisada. Puede incorporar
    #   aristas teacher para evitar targets bloqueados cuando se usa KNN.
    # ------------------------------------------------------------------
    base_mask = sinkhorn_mask.float()

    decoder_mask = base_mask.clone()
    loss_mask = base_mask.clone()

    if (
        isinstance(batch.get("teacher", None), dict)
        and KEY_TOUR_NEXT in batch["teacher"]
        and batch["teacher"][KEY_TOUR_NEXT] is not None
    ):
        tour_next = batch["teacher"][KEY_TOUR_NEXT].to(device).long()

        if tour_next.ndim != 2:
            error_msg = f"tour_next debe ser (B,N). Recibido {tuple(tour_next.shape)}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
        row_idx = torch.arange(N, device=device).view(1, N).expand(B, N)

        valid_next = (tour_next >= 0) & (tour_next < N)

        # Unión solo para la loss: loss_mask = base_mask ∪ teacher_edges.
        loss_mask[batch_idx[valid_next], row_idx[valid_next], tour_next[valid_next]] = 1.0

        # Por seguridad, respeta allow_self si se pide explícitamente no permitir diagonal.
        if not allow_self:
            idx = torch.arange(N, device=device)
            loss_mask[:, idx, idx] = 0.0
    
    batch["inputs"][KEY_COORDS_NORM] = coords
    batch["inputs"]["node_mask"] = node_mask
    batch["inputs"]["item_mask"] = item_mask

    # Máscara del modelo/Sinkhorn: no contiene teacher edges añadidas.
    batch["inputs"][KEY_SINKHORN_MASK] = base_mask

    # Máscara para decodificación: por ahora coincide con la máscara del modelo.
    batch["inputs"][KEY_DECODER_MASK] = decoder_mask

    # Máscara para loss supervisada: puede contener teacher edges añadidas.
    batch["inputs"][KEY_LOSS_MASK] = loss_mask

    if compute_tour_adj:
        batch["teacher"][KEY_TOUR_ADJ] = build_tour_adj_from_next(
            batch["teacher"][KEY_TOUR_NEXT].to(device).long()
        )

    return batch