#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/hpo/sampling.py

"""Muestreo de parámetros para trials de Optuna.

Este módulo transforma un trial de Optuna en configuraciones completas del
modelo y del entrenamiento usadas por el workflow HPO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ttp_packages.hpo.config import (
    OptunaModelSearchSpace,
    OptunaTrainSearchSpace,
)
from src.ttp_packages.modeling.config import TTPArchitectureParams
from src.ttp_packages.training.config import TrainingParams

if TYPE_CHECKING:
    import optuna


def _sample_model_params(
    trial: optuna.trial.Trial,
    model_space: OptunaModelSearchSpace,
) -> TTPArchitectureParams:
    """Muestrea una configuración completa de arquitectura para un trial.

    Además de los hiperparámetros clásicos del Transformer, esta función
    samplea las opciones nuevas de representación:

    - ``node_feature_set``:
      ``"basic"`` mantiene compatibilidad con modelos antiguos;
      ``"ttp_v1"`` activa features TTP-aware de ciudad.
    - ``edge_feature_mode``:
      ``"none"`` mantiene el comportamiento anterior;
      ``"distance_v1"`` activa atributos simples de arista basados en distancia.

    Args:
        trial: Trial actual de Optuna.
        model_space: Search space de arquitectura.

    Returns:
        Instancia completa de ``TTPArchitectureParams`` con arquitectura,
        representación de nodos y modo de atributos de arista.

    Raises:
        ValueError: Si no existe ningún ``n_heads`` compatible con ``d_model``.
    """
    d_model = trial.suggest_categorical(
        "d_model",
        list(model_space.d_model_choices),
    )

    # Multi-head attention requiere que d_model sea divisible por n_heads.
    valid_n_heads = [
        n_heads
        for n_heads in model_space.n_heads_choices
        if d_model % n_heads == 0
    ]

    if not valid_n_heads:
        raise ValueError(
            f"No hay n_heads válidos para d_model={d_model}. "
            f"Candidates={model_space.n_heads_choices}"
        )

    # Estos dos campos permiten comparar modo antiguo vs. modo enriquecido sin
    # cambiar el formato persistente del dataset.
    node_feature_set = trial.suggest_categorical(
        "node_feature_set",
        list(model_space.node_feature_set_choices),
    )
    edge_feature_mode = trial.suggest_categorical(
        "edge_feature_mode",
        list(model_space.edge_feature_mode_choices),
    )

    return TTPArchitectureParams(
        d_model=d_model,
        n_heads=trial.suggest_categorical("n_heads", list(valid_n_heads)),
        n_layers=trial.suggest_categorical(
            "n_layers",
            list(model_space.n_layers_choices),
        ),
        d_ff=trial.suggest_categorical(
            "d_ff",
            list(model_space.d_ff_choices),
        ),
        dropout=trial.suggest_float(
            "dropout",
            model_space.dropout_low,
            model_space.dropout_high,
        ),
        sinkhorn_iter=trial.suggest_categorical(
            "sinkhorn_iter",
            list(model_space.sinkhorn_iter_choices),
        ),
        sink_tau=trial.suggest_float(
            "sink_tau",
            model_space.sink_tau_low,
            model_space.sink_tau_high,
        ),
        coupling_iters=trial.suggest_categorical(
            "coupling_iters",
            list(model_space.coupling_iters_choices),
        ),
        node_feature_set=node_feature_set,
        edge_feature_mode=edge_feature_mode,
    )

def _sample_train_params(
    trial: optuna.trial.Trial,
    train_space: OptunaTrainSearchSpace,
) -> TrainingParams:
    """Muestrea una configuración completa de entrenamiento para un trial.

    Esta función samplea entrenamiento tour-only sin ejecutar solvers completos
    dentro del objective. La métrica objetivo sigue siendo entrenamiento/
    validación del modelo neuronal.

    Reglas nuevas:
    - ``sinkhorn_nll_weight`` se samplea desde el search space de entrenamiento.
      ``0.0`` mantiene el comportamiento anterior.
    - ``use_separate_loss_mask`` se registra en Optuna como metadata/restricción,
      pero no se pasa a ``TrainingParams`` porque la separación de máscaras ya
      está fija en el pipeline.
    - si el trial eligió ``edge_feature_mode == "distance_v1"`` en
      ``_sample_model_params()``, entonces ``compute_dist_matrix`` se fuerza a
      ``True`` para que el modelo pueda construir atributos de arista.

    Args:
        trial: Trial actual de Optuna.
        train_space: Search space de entrenamiento.

    Returns:
        Instancia completa de ``TrainingParams`` coherente con la arquitectura
        sampleada para el mismo trial.
    """
    mask_mode = trial.suggest_categorical(
        "mask_mode",
        list(train_space.mask_mode_choices),
    )

    # knn_k solo afecta al modo KNN; en modo dense se conserva un valor válido
    # para que TrainingParams siempre tenga configuración completa.
    knn_k = int(train_space.knn_k_default)
    if mask_mode == "knn":
        knn_k = trial.suggest_categorical(
            "knn_k",
            list(train_space.knn_k_choices),
        )

    # _sample_model_params() ya registró edge_feature_mode en trial.params.
    # Si se eligió distance_v1, la matriz de distancias es obligatoria.
    edge_feature_mode = str(trial.params.get("edge_feature_mode", "none"))

    resolved_compute_dist_matrix = bool(train_space.compute_dist_matrix)
    if edge_feature_mode == "distance_v1":
        resolved_compute_dist_matrix = True

    # Metadata/restricción del estudio. No se pasa a TrainingParams porque la
    # separación sinkhorn_mask/loss_mask/decoder_mask ya está fija en el código.
    trial.suggest_categorical(
        "use_separate_loss_mask",
        list(train_space.use_separate_loss_mask_choices),
    )

    sinkhorn_nll_weight = float(
        trial.suggest_categorical(
            "sinkhorn_nll_weight",
            list(train_space.sinkhorn_nll_weight_choices),
        )
    )

    return TrainingParams(
        # Reproducibilidad y split.
        seed=int(train_space.seed),
        val_frac=float(train_space.val_frac),

        # DataLoader.
        batch_size=trial.suggest_categorical(
            "batch_size",
            list(train_space.batch_size_choices),
        ),
        shuffle_train=bool(train_space.shuffle_train),
        num_workers=int(train_space.num_workers),

        # Optimización.
        epochs=int(train_space.epochs),
        lr=trial.suggest_float(
            "lr",
            train_space.lr_low,
            train_space.lr_high,
            log=train_space.lr_log,
        ),
        weight_decay=trial.suggest_float(
            "weight_decay",
            train_space.weight_decay_low,
            train_space.weight_decay_high,
            log=train_space.weight_decay_log,
        ),
        grad_clip_norm=trial.suggest_categorical(
            "grad_clip_norm",
            list(train_space.grad_clip_norm_choices),
        ),
        use_amp=trial.suggest_categorical(
            "use_amp",
            list(train_space.use_amp_choices),
        ),

        # Tour y máscaras.
        mask_mode=mask_mode,
        knn_k=int(knn_k),
        allow_self=bool(train_space.allow_self),
        sym=bool(train_space.sym),
        compute_dist_matrix=resolved_compute_dist_matrix,
        apply_mask_to_loss=trial.suggest_categorical(
            "apply_mask_to_loss",
            list(train_space.apply_mask_to_loss_choices),
        ),
        sinkhorn_nll_weight=sinkhorn_nll_weight,

        # Device.
        device=train_space.device,

        # Logging y early stopping.
        verbose=bool(train_space.verbose),
        patience=int(train_space.patience),
        min_delta=float(train_space.min_delta),
        overfit_patience=(
            None
            if train_space.overfit_patience is None
            else int(train_space.overfit_patience)
        ),
        overfit_min_delta=float(train_space.overfit_min_delta),
    )
