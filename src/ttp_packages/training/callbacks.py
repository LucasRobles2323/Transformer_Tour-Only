#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/training/callbacks.py

"""Callbacks y estado auxiliar para el entrenamiento.

Este módulo concentra el estado mutable del entrenamiento, actualización de
mejores métricas, early stopping, restauración de pesos y construcción del
summary final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrainingState:
    """Estado acumulado durante el entrenamiento.

    Attributes:
        best_val_loss: Mejor pérdida de validación observada.
        best_val_acc_at_best_loss: Accuracy de validación en la mejor pérdida.
        best_val_acc_seen: Mejor accuracy de validación observada.
        best_train_loss: Mejor pérdida de entrenamiento observada.
        best_train_acc: Mejor accuracy de entrenamiento observada.
        best_epoch: Época asociada al mejor ``val_loss``.
        best_state_dict: Copia en CPU del mejor ``state_dict``.
        patience_counter_val_loss: Épocas consecutivas sin mejora de ``val_loss``.
        overfit_counter: Pasos consecutivos con patrón de sobreajuste.
        prev_train_loss: Pérdida de entrenamiento de la época anterior.
        prev_val_loss: Pérdida de validación de la época anterior.
        stop_reason: Motivo de detención del entrenamiento.
        history: Historial de métricas por época.
    """

    # Métrica principal para seleccionar checkpoint: menor val_loss.
    best_val_loss: float = float("inf")
    best_val_acc_at_best_loss: float = 0.0

    # Métrica informativa: mejor val_acc vista, aunque no coincida con best_val_loss.
    best_val_acc_seen: float = 0.0

    # Métricas informativas de entrenamiento.
    best_train_loss: float = float("inf")
    best_train_acc: float = 0.0

    # Epoch y pesos asociados al mejor val_loss.
    best_epoch: int = -1
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    # Cuenta épocas consecutivas sin mejora suficiente de val_loss.
    patience_counter_val_loss: int = 0

    # Cuenta épocas consecutivas donde train_loss mejora pero val_loss empeora.
    overfit_counter: int = 0
    prev_train_loss: Optional[float] = None
    prev_val_loss: Optional[float] = None

    stop_reason: str = "completed_all_epochs"
    history: List[Dict[str, Any]] = field(default_factory=list)


def append_epoch_history(
    state: TrainingState,
    *,
    epoch: int,
    lr: float,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
) -> None:
    """Agrega las métricas de una época al historial.

    Args:
        state: Estado de entrenamiento a actualizar.
        epoch: Número de época.
        lr: Learning rate actual.
        train_loss: Pérdida promedio de entrenamiento.
        train_acc: Accuracy promedio de entrenamiento.
        val_loss: Pérdida promedio de validación.
        val_acc: Accuracy promedio de validación.
    """
    # Se fuerza conversión a tipos simples para que el historial sea serializable.
    state.history.append(
        {
            "epoch": int(epoch),
            "lr": float(lr),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
    )


def update_training_state(
    state: TrainingState,
    *,
    model: torch.nn.Module,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    min_delta: float,
    overfit_min_delta: float,
) -> None:
    """Actualiza mejores métricas, checkpoint y contadores de parada.

    Args:
        state: Estado de entrenamiento a modificar.
        model: Modelo entrenado en la época actual.
        epoch: Época actual.
        train_loss: Pérdida promedio de entrenamiento.
        train_acc: Accuracy promedio de entrenamiento.
        val_loss: Pérdida promedio de validación.
        val_acc: Accuracy promedio de validación.
        min_delta: Mejora mínima para considerar avance real.
        overfit_min_delta: Cambio mínimo para detectar patrón de sobreajuste.
    """
    # Normaliza entradas a float para evitar mezclar tensores/NumPy/scalars.
    train_loss = float(train_loss)
    train_acc = float(train_acc)
    val_loss = float(val_loss)
    val_acc = float(val_acc)
    min_delta = float(min_delta)
    overfit_min_delta = float(overfit_min_delta)

    # Métricas de train son informativas: no controlan checkpoint ni early stopping.
    state.best_train_loss = min(state.best_train_loss, train_loss)
    state.best_train_acc = max(state.best_train_acc, train_acc)

    # val_loss controla el mejor checkpoint y el early stopping principal.
    improved_val_loss = val_loss < (state.best_val_loss - min_delta)

    # val_acc se guarda como métrica informativa independiente.
    improved_val_acc = val_acc > (state.best_val_acc_seen + min_delta)

    if improved_val_loss:
        state.best_val_loss = val_loss
        state.best_val_acc_at_best_loss = val_acc
        state.best_epoch = int(epoch)

        # Guarda el mejor checkpoint en CPU para restaurarlo sin depender del device.
        state.best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        # Al mejorar val_loss, se reinicia la paciencia del early stopping principal.
        state.patience_counter_val_loss = 0
    else:
        # Si val_loss no mejora lo suficiente, aumenta el contador de paciencia.
        state.patience_counter_val_loss += 1

    if improved_val_acc:
        state.best_val_acc_seen = val_acc

    # Sobreajuste simple: train_loss mejora, pero val_loss empeora.
    # Esto detecta divergencia entre entrenamiento y validación de forma consecutiva.
    if state.prev_train_loss is not None and state.prev_val_loss is not None:
        train_improved_step = train_loss < (state.prev_train_loss - overfit_min_delta)
        val_worsened_step = val_loss > (state.prev_val_loss + overfit_min_delta)

        if train_improved_step and val_worsened_step:
            state.overfit_counter += 1
        else:
            # Si el patrón se rompe, reiniciamos el contador de overfitting.
            state.overfit_counter = 0

    # Se guardan las pérdidas actuales para comparar contra la próxima época.
    state.prev_train_loss = train_loss
    state.prev_val_loss = val_loss


def should_stop_early(
    state: TrainingState,
    *,
    patience: int,
    overfit_patience: Optional[int] = None,
) -> bool:
    """Indica si el entrenamiento debe detenerse.

    Args:
        state: Estado actual del entrenamiento.
        patience: Máximo de épocas sin mejora de ``val_loss``.
        overfit_patience: Máximo de pasos consecutivos con patrón de sobreajuste.

    Returns:
        ``True`` si corresponde detener el entrenamiento.
    """
    # Criterio 1: no hubo mejora suficiente de val_loss durante varias épocas.
    if state.patience_counter_val_loss >= int(patience):
        state.stop_reason = "early_stop_validation_no_loss_improvement"
        return True

    # Criterio 2: patrón consecutivo de overfitting.
    if overfit_patience is not None and state.overfit_counter >= int(overfit_patience):
        state.stop_reason = "early_stop_overfitting_detected"
        return True

    return False


def restore_best_weights(
    model: torch.nn.Module,
    state: TrainingState,
) -> bool:
    """Restaura los mejores pesos guardados en el estado.

    Args:
        model: Modelo a restaurar.
        state: Estado con el mejor ``state_dict``.

    Returns:
        ``True`` si se restauraron pesos; ``False`` si no había checkpoint.
    """
    if state.best_state_dict is not None:
        # Restaura el modelo asociado al menor val_loss observado.
        model.load_state_dict(state.best_state_dict)
        return True

    return False


def build_training_summary(
    state: TrainingState,
    *,
    n_total: int,
    n_train: int,
    n_val: int,
    restored_best_weights: bool,
) -> Dict[str, Any]:
    """Construye un resumen final del entrenamiento.

    Args:
        state: Estado final del entrenamiento.
        n_total: Número total de samples.
        n_train: Número de samples de entrenamiento.
        n_val: Número de samples de validación.
        restored_best_weights: Si es True, se restauraron los mejores pesos.

    Returns:
        Diccionario serializable con métricas y metadata del entrenamiento.
    """
    # El summary queda listo para exportarse a JSON junto con el checkpoint.
    return {
        "stop_reason": state.stop_reason,
        "epochs_ran": len(state.history),
        "best_epoch": state.best_epoch,
        "best_val_loss": state.best_val_loss,
        "best_val_acc_at_best_loss": state.best_val_acc_at_best_loss,
        "best_val_acc_seen": state.best_val_acc_seen,
        "best_train_loss": state.best_train_loss,
        "best_train_acc": state.best_train_acc,
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "restored_best_weights": bool(restored_best_weights),
    }