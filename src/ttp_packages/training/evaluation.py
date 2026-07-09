#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/training/evaluation.py

"""Evaluación del modelo durante entrenamiento."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.ttp_packages.ml_data.torch.transforms.augment import augment_batch_on_device
from src.ttp_packages.training.losses import compute_tour_loss_and_acc
from src.ttp_packages.training.utils import move_to_device


@torch.no_grad()
def eval_tour_only(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    mask_mode: str,
    knn_k: int,
    allow_self: bool,
    sym: bool,
    compute_dist_matrix: bool,
    apply_mask_to_loss: bool,
    sinkhorn_nll_weight: float = 0.0,
) -> Tuple[float, float]:
    """Evalúa el modelo en modo tour-only.

    Args:
        model: Modelo PyTorch a evaluar.
        loader: DataLoader de validación.
        device: Dispositivo de evaluación.
        mask_mode: Modo de máscara usado por augmentación.
        knn_k: Número de vecinos para máscara KNN.
        allow_self: Si es True, permite auto-conexiones.
        sym: Si es True, fuerza simetría.
        compute_dist_matrix: Si es True, calcula matriz de distancias.
        apply_mask_to_loss: Si es True, aplica máscara al cálculo de pérdida.
        sinkhorn_nll_weight: Peso de la pérdida auxiliar NLL sobre Sinkhorn. Si es
            ``0.0``, se desactiva y se conserva el comportamiento anterior.

    Returns:
        Tupla ``(mean_loss, mean_acc)`` con pérdida y accuracy promedio.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0

    for batch in loader:
        # Garantiza que todos los tensores del batch estén en el mismo device.
        batch = move_to_device(batch, device)

        # Reconstruye features derivadas y máscaras necesarias antes del forward.
        # Esto mantiene liviano el dataset y calcula distancias/máscaras on-the-fly.
        batch = augment_batch_on_device(
            batch,
            mask_mode=mask_mode,
            knn_k=knn_k,
            allow_self=allow_self,
            sym=sym,
            compute_dist_matrix=compute_dist_matrix,
            compute_tour_adj=False,
        )

        inputs = batch["inputs"]
        teacher = batch["teacher"]

        # En evaluación se conservan ambas salidas para mantener el mismo contrato que
        # en entrenamiento. En este paso la loss usa edge_logits; transition_probs se
        # usará luego para la NLL auxiliar opcional sobre Sinkhorn.
        transition_probs, edge_logits = model(
            coords=inputs["coords"],
            W=inputs["W"],
            node_mask=inputs["node_mask"],
            sinkhorn_mask=inputs["sinkhorn_mask"],
            item_city=inputs["item_city"],
            item_profit=inputs["item_profit"],
            item_weight=inputs["item_weight"],
            item_mask=inputs["item_mask"],
            min_speed=inputs["min_speed"],
            max_speed=inputs["max_speed"],
            rent_per_time=inputs["rent_per_time"],
            dist_matrix=inputs.get("dist_matrix", None),
        )

        tour_next = teacher["tour_next"].long()
        node_mask = inputs["node_mask"].float()

        # Igual que en entrenamiento: la CE usa loss_mask si existe.
        # Si el batch viene de un flujo antiguo, se usa sinkhorn_mask como fallback.
        mask_for_loss = inputs.get("loss_mask", inputs.get("sinkhorn_mask", None))

        loss, correct, count = compute_tour_loss_and_acc(
            edge_logits=edge_logits,
            tour_next=tour_next,
            node_mask=node_mask,
            sinkhorn_mask=mask_for_loss,
            apply_mask_to_loss=apply_mask_to_loss,
            transition_probs=transition_probs,
            sinkhorn_nll_weight=sinkhorn_nll_weight,
            sinkhorn_nll_mask=inputs.get("sinkhorn_mask", None),
        )

        # Se pondera por count para promediar por nodo válido, no por batch.
        count_value = float(count.detach().item())
        total_loss += float(loss.detach().item()) * count_value
        total_correct += float(correct.detach().item())
        total_count += count_value

    mean_loss = total_loss / max(total_count, 1.0)
    mean_acc = total_correct / max(total_count, 1.0)

    return mean_loss, mean_acc