#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/training/losses.py

"""Funciones de pérdida y métricas para entrenamiento tour-only."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


LOSS_REDUCTION_NONE = "none"


def compute_tour_loss_and_acc(
    edge_logits: torch.Tensor,
    tour_next: torch.Tensor,
    node_mask: torch.Tensor,
    sinkhorn_mask: Optional[torch.Tensor] = None,
    apply_mask_to_loss: bool = True,
    transition_probs: Optional[torch.Tensor] = None,
    sinkhorn_nll_weight: float = 0.0,
    sinkhorn_nll_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcula pérdida y exactitud para supervisión tour-only.

    La pérdida principal es cross-entropy sobre ``edge_logits`` contra
    ``tour_next``. Opcionalmente agrega una NLL auxiliar sobre
    ``transition_probs`` para supervisar directamente la matriz suave producida
    por Sinkhorn.

    Importante:
        - ``sinkhorn_mask`` se usa para enmascarar la CE. En el flujo nuevo,
          normalmente corresponde a ``loss_mask``.
        - ``sinkhorn_nll_mask`` se usa para decidir en qué aristas teacher se
          puede aplicar la NLL sobre Sinkhorn. Debe corresponder a la máscara
          real del modelo, es decir, ``sinkhorn_mask`` original, no ``loss_mask``.

    Args:
        edge_logits: Logits por arista ``i -> j`` con shape ``(B, N, N)``.
        tour_next: Target supervisado con shape ``(B, N)``. Cada posición indica
            la siguiente ciudad esperada para el nodo origen.
        node_mask: Máscara de nodos válidos con shape ``(B, N)``.
        sinkhorn_mask: Máscara usada para la CE. En entrenamiento nuevo debe ser
            ``loss_mask`` para permitir supervisión de teacher edges fuera del KNN.
        apply_mask_to_loss: Si es True, bloquea logits no permitidos antes de CE.
        transition_probs: Matriz suave producida por Sinkhorn con shape
            ``(B, N, N)``. Se usa solo si ``sinkhorn_nll_weight > 0``.
        sinkhorn_nll_weight: Peso del término auxiliar NLL sobre Sinkhorn.
        sinkhorn_nll_mask: Máscara real del modelo/Sinkhorn. La NLL auxiliar solo
            se aplica a teacher edges permitidas por esta máscara.
        eps: Valor pequeño para estabilidad numérica.

    Returns:
        Tupla ``(loss, correct, count)``:
            - ``loss``: pérdida promedio por nodo válido.
            - ``correct``: cantidad de aciertos.
            - ``count``: cantidad de nodos válidos usados para la métrica.

    Raises:
        ValueError: Si se pide NLL auxiliar pero no se entrega
            ``transition_probs``.
    """
    if edge_logits.ndim != 3:
        raise ValueError(
            f"edge_logits debe tener shape (B,N,N). Recibido {tuple(edge_logits.shape)}."
        )

    if tour_next.ndim != 2:
        raise ValueError(
            f"tour_next debe tener shape (B,N). Recibido {tuple(tour_next.shape)}."
        )

    batch_size, n_cities, n_cols = edge_logits.shape
    if n_cities != n_cols:
        raise ValueError(
            f"edge_logits debe ser cuadrado en N. Recibido {tuple(edge_logits.shape)}."
        )

    if tour_next.shape != (batch_size, n_cities):
        raise ValueError(
            "tour_next no coincide con edge_logits. "
            f"tour_next={tuple(tour_next.shape)}, edge_logits={tuple(edge_logits.shape)}."
        )

    node_mask = node_mask.float().to(edge_logits.device)
    targets = tour_next.long().to(edge_logits.device)

    logits_for_ce = edge_logits

    # ------------------------------------------------------------------
    # 1) CE supervisada sobre edge_logits.
    # ------------------------------------------------------------------
    if apply_mask_to_loss and sinkhorn_mask is not None:
        ce_mask = sinkhorn_mask.to(device=edge_logits.device)

        if ce_mask.dim() == 2:
            ce_mask = ce_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif ce_mask.dim() == 3 and ce_mask.size(0) != batch_size:
            ce_mask = ce_mask[:1].expand(batch_size, -1, -1)

        ce_mask = ce_mask[:, :n_cities, :n_cities].float()

        neg_inf = torch.finfo(edge_logits.dtype).min
        logits_for_ce = edge_logits.masked_fill(ce_mask <= 0.0, neg_inf)

    ce_per_node = F.cross_entropy(
        logits_for_ce.reshape(batch_size * n_cities, n_cities),
        targets.reshape(batch_size * n_cities),
        reduction=LOSS_REDUCTION_NONE,
    ).view(batch_size, n_cities)

    valid_node_mask = node_mask
    count = valid_node_mask.sum().clamp_min(1.0)

    ce_loss = (ce_per_node * valid_node_mask).sum() / count
    loss = ce_loss

    # ------------------------------------------------------------------
    # 2) NLL auxiliar sobre transition_probs.
    # ------------------------------------------------------------------
    weight = float(sinkhorn_nll_weight)

    if weight > 0.0:
        if transition_probs is None:
            raise ValueError(
                "transition_probs es obligatorio cuando sinkhorn_nll_weight > 0."
            )

        if transition_probs.shape != edge_logits.shape:
            raise ValueError(
                "transition_probs debe tener la misma shape que edge_logits. "
                f"transition_probs={tuple(transition_probs.shape)}, "
                f"edge_logits={tuple(edge_logits.shape)}."
            )

        probs = transition_probs.to(device=edge_logits.device).float()
        target_probs = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        nll_per_node = -torch.log(target_probs.clamp_min(float(eps)))

        nll_valid_mask = valid_node_mask

        if sinkhorn_nll_mask is not None:
            model_mask = sinkhorn_nll_mask.to(device=edge_logits.device)

            if model_mask.dim() == 2:
                model_mask = model_mask.unsqueeze(0).expand(batch_size, -1, -1)
            elif model_mask.dim() == 3 and model_mask.size(0) != batch_size:
                model_mask = model_mask[:1].expand(batch_size, -1, -1)

            model_mask = model_mask[:, :n_cities, :n_cities].float()

            # La NLL auxiliar solo aprende desde teacher edges permitidas por la
            # máscara real del modelo. Esto evita usar loss_mask aquí.
            teacher_allowed = model_mask.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            nll_valid_mask = nll_valid_mask * (teacher_allowed > 0.0).float()

        nll_count = nll_valid_mask.sum().clamp_min(1.0)
        sinkhorn_nll = (nll_per_node * nll_valid_mask).sum() / nll_count

        loss = loss + (weight * sinkhorn_nll)

    # ------------------------------------------------------------------
    # 3) Accuracy sobre logits.
    # ------------------------------------------------------------------
    pred_next = logits_for_ce.argmax(dim=-1)
    correct = ((pred_next == targets).float() * valid_node_mask).sum()

    return loss, correct, count