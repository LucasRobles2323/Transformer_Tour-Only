#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/training/config.py

"""Parámetros de configuración para entrenamiento tour-only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class TrainingParams:
    """Parámetros de entrenamiento del modelo tour-only.

    Attributes:
        seed: Semilla para reproducibilidad.
        val_frac: Fracción del dataset destinada a validación.
        batch_size: Tamaño de batch.
        shuffle_train: Si es True, baraja el subset de entrenamiento.
        num_workers: Número de workers del ``DataLoader``.
        epochs: Cantidad máxima de épocas.
        lr: Learning rate del optimizador.
        weight_decay: Decaimiento de pesos.
        grad_clip_norm: Norma máxima para recorte de gradientes.
        use_amp: Activa/desactiva Automatic Mixed Precision. Si es ``None``,
            se activa automáticamente solo en CUDA.
        mask_mode: Modo de máscara usado por la augmentación.
        knn_k: Número de vecinos para máscaras KNN.
        allow_self: Si es True, permite auto-conexiones.
        sym: Si es True, fuerza simetría en máscaras.
        compute_dist_matrix: Si es True, calcula matriz de distancias.
        apply_mask_to_loss: Si es True, aplica máscara al cálculo de pérdida.
        sinkhorn_nll_weight: Peso de la pérdida auxiliar NLL aplicada sobre
            ``transition_probs``. Si es ``0.0``, se desactiva y el entrenamiento
            conserva el comportamiento anterior.
        device: Dispositivo de entrenamiento. Si es ``None``, se infiere.
        verbose: Si es True, imprime logs por época.
        patience: Épocas sin mejora de ``val_loss`` antes de detener.
        min_delta: Mejora mínima para resetear paciencia.
        overfit_patience: Pasos consecutivos de sobreajuste antes de detener.
        overfit_min_delta: Cambio mínimo para detectar sobreajuste.
    """

    # Reproducibilidad y split train/validation.
    seed: int = 123
    val_frac: float = 0.20

    # Parámetros del DataLoader.
    batch_size: int = 16
    shuffle_train: bool = True
    num_workers: int = 0

    # Parámetros de optimización.
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 3e-4
    grad_clip_norm: float = 1.0

    # Si es None, se activa AMP automáticamente solo cuando el device es CUDA.
    use_amp: Optional[bool] = None

    # Configuración de máscaras y features reconstruidas por augment_batch_on_device().
    mask_mode: str = "dense"
    knn_k: int = 10
    allow_self: bool = False
    sym: bool = True
    compute_dist_matrix: bool = True
    apply_mask_to_loss: bool = True

    # Peso de la pérdida auxiliar sobre la matriz suave producida por Sinkhorn.
    # Con 0.0 se conserva exactamente el comportamiento anterior.
    sinkhorn_nll_weight: float = 0.0

    # Si es None, engine.py infiere el device desde el modelo o desde runtime.
    device: Optional[Union[str, torch.device]] = None

    # Logging y early stopping principal por falta de mejora en val_loss.
    verbose: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Stop explícito por overfitting. Usa None para desactivar este criterio.
    overfit_patience: Optional[int] = 5
    overfit_min_delta: float = 1e-4