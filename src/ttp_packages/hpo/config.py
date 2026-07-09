#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/hpo/config.py

"""Configuración de HPO/Optuna para modelos TTP tour-only.

Este módulo concentra los parámetros operativos del estudio, los espacios de
búsqueda de arquitectura y entrenamiento, los defaults compartidos y la
validación básica del workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from src.ttp_packages.infrastructure.storage.paths import build_optuna_db_path


@dataclass
class OptunaWorkflowParams:
    """Parámetros operativos del estudio y de su persistencia.

    Attributes:
        study_name: Nombre lógico del estudio. También determina el nombre del
            archivo SQLite.
        direction: Dirección de optimización de Optuna.
        load_if_exists: Si es ``True``, reutiliza un estudio existente.
        n_trials: Número total de trials a lanzar.
        n_jobs: Cantidad de jobs concurrentes.
        sampler_seed: Semilla del sampler TPE.
        gc_after_trial: Si es ``True``, fuerza GC al terminar cada trial.
        heartbeat_interval: Heartbeat en segundos para detección de stale trials.
        grace_period: Período de gracia antes de marcar un trial como stale.
        timeout_days: Días del límite de tiempo total del estudio.
        timeout_hours: Horas del límite de tiempo total del estudio.
        timeout_minutes: Minutos del límite de tiempo total del estudio.
        timeout_seconds: Segundos del límite de tiempo total del estudio.
        save_best_artifacts: Si es ``True``, permite exportar artefactos del
            mejor trial.
        save_best_only_if_improved: Si es ``True``, exporta artefactos solo
            cuando mejora el mejor resultado persistido.
        export_best_from_current_run: Si es ``True``, exporta el mejor trial de
            la corrida actual.
        retrain_best_at_end: Si es ``True``, reentrena el mejor trial al final
            del estudio.
        artifact_min_delta: Mejora mínima requerida para considerar un artefacto
            como mejor.
        save_top_k: Cantidad de mejores trials cuyos artefactos se conservarán.
        fail_running_on_start: Si es ``True``, marca como fallidos los trials que
            quedaron en estado running al iniciar.
        verbose: Si es ``True``, habilita logging más detallado.
    """

    study_name: str = "tour_hpo"
    direction: str = "minimize"
    load_if_exists: bool = True
    n_trials: int = 50
    n_jobs: int = 1
    sampler_seed: Optional[int] = 123
    gc_after_trial: bool = True
    heartbeat_interval: Optional[int] = None
    grace_period: Optional[int] = None

    # Límite de tiempo opcional para el estudio.
    # Se puede usar uno solo o combinarlos, por ejemplo:
    # timeout_days=1 y timeout_hours=6 equivale a 30 horas.
    timeout_days: Optional[float] = None
    timeout_hours: Optional[float] = None
    timeout_minutes: Optional[float] = None
    timeout_seconds: Optional[float] = None

    # Exportación de artefactos.
    save_best_artifacts: bool = True
    save_best_only_if_improved: bool = True
    export_best_from_current_run: bool = True
    retrain_best_at_end: bool = False
    artifact_min_delta: float = 0.0

    save_top_k: int = 1
    fail_running_on_start: bool = True
    verbose: bool = True

    @property
    def effective_timeout_seconds(self) -> Optional[float]:
        """Calcula el timeout total del estudio en segundos.

        Returns:
            Duración máxima en segundos, o ``None`` si no se configuró timeout.
        """
        total = 0.0

        if self.timeout_days is not None:
            total += float(self.timeout_days) * 24.0 * 3600.0

        if self.timeout_hours is not None:
            total += float(self.timeout_hours) * 3600.0

        if self.timeout_minutes is not None:
            total += float(self.timeout_minutes) * 60.0

        if self.timeout_seconds is not None:
            total += float(self.timeout_seconds)

        return total if total > 0.0 else None

    @property
    def storage_path(self) -> Path:
        """Construye la ruta del archivo SQLite del estudio.

        Returns:
            Ruta al archivo ``.db`` usado por Optuna.
        """
        return build_optuna_db_path(self.study_name)

    @property
    def storage_url(self) -> str:
        """Construye la URL de storage que consume Optuna.

        Returns:
            URL SQLite con el prefijo ``sqlite:///``.
        """
        return f"sqlite:///{self.storage_path.as_posix()}"



@dataclass
class OptunaModelSearchSpace:
    """Search space del modelo usado por Optuna.

    Esta clase describe qué campos de ``TTPArchitectureParams`` son explorables y
    cómo Optuna debe muestrearlos. Incluye tanto parámetros clásicos de la
    arquitectura Transformer como opciones nuevas de representación TTP-aware.

    Las opciones nuevas permiten comparar:

    - modo compatible antiguo:
      ``node_feature_set="basic"`` y ``edge_feature_mode="none"``;
    - modo enriquecido:
      ``node_feature_set="ttp_v1"`` y ``edge_feature_mode="distance_v1"``.

    Attributes:
        d_model_choices: Opciones para la dimensión interna del modelo.
        n_heads_choices: Opciones para número de cabezas de atención.
        n_layers_choices: Opciones para profundidad del encoder.
        d_ff_choices: Opciones para la dimensión de la red feed-forward.
        dropout_low: Cota inferior de dropout.
        dropout_high: Cota superior de dropout.
        sinkhorn_iter_choices: Opciones para iteraciones de Sinkhorn.
        sink_tau_low: Cota inferior de tau.
        sink_tau_high: Cota superior de tau.
        coupling_iters_choices: Opciones para coupling_iters.
        node_feature_set_choices: Opciones de representación por ciudad.
            ``"basic"`` conserva el comportamiento antiguo. ``"ttp_v1"``
            agrega features TTP-aware derivadas de profit, peso, densidad,
            capacidad y presencia de ítems.
        edge_feature_mode_choices: Opciones de atributos de arista.
            ``"none"`` conserva el comportamiento antiguo. ``"distance_v1"``
            agrega atributos simples basados en distancia, velocidad y renta.
            Cuando se usa ``"distance_v1"``, el muestreo de entrenamiento debe
            forzar ``compute_dist_matrix=True``.
    """

    d_model_choices: Sequence[int] = (32, 64, 128, 256)
    n_heads_choices: Sequence[int] = (2, 4, 8)
    n_layers_choices: Sequence[int] = (2, 3, 4, 6)
    d_ff_choices: Sequence[int] = (128, 256, 512)
    dropout_low: float = 0.0
    dropout_high: float = 0.30
    sinkhorn_iter_choices: Sequence[int] = (40, 80, 120, 160)
    sink_tau_low: float = 0.05
    sink_tau_high: float = 0.30
    coupling_iters_choices: Sequence[int] = (1,)
    node_feature_set_choices: Sequence[str] = ("basic", "ttp_v1")
    edge_feature_mode_choices: Sequence[str] = ("none", "distance_v1")


@dataclass
class OptunaTrainSearchSpace:
    """Search space completo para ``TrainingParams``.

    Regla de diseño:
    - todo campo de ``TrainingParams`` debe existir aquí;
    - algunos campos son explorables mediante ``trial.suggest_*``;
    - otros permanecen fijos para todos los trials del estudio.

    Esta configuración también incluye los campos nuevos de entrenamiento:

    - ``sinkhorn_nll_weight_choices`` permite activar o desactivar la pérdida
      auxiliar sobre ``transition_probs``.
    - ``use_separate_loss_mask_choices`` se conserva como metadata/restricción
      del estudio. No se pasa a ``TrainingParams`` porque la separación entre
      ``sinkhorn_mask``, ``loss_mask`` y ``decoder_mask`` ya está implementada
      de forma fija en el pipeline.

    Attributes:
        seed: Semilla de reproducibilidad del entrenamiento.
        val_frac: Fracción del dataset usada para validación.
        batch_size_choices: Opciones de tamaño de batch.
        shuffle_train: Si es ``True``, mezcla el split de entrenamiento.
        num_workers: Workers del DataLoader.
        epochs: Número fijo de épocas por trial.
        lr_low: Cota inferior del learning rate.
        lr_high: Cota superior del learning rate.
        lr_log: Si es ``True``, muestrea ``lr`` en escala logarítmica.
        weight_decay_low: Cota inferior de weight decay.
        weight_decay_high: Cota superior de weight decay.
        weight_decay_log: Si es ``True``, muestrea ``weight_decay`` en log.
        grad_clip_norm_choices: Opciones para gradient clipping.
        use_amp_choices: Opciones para AMP.
        mask_mode_choices: Opciones del modo de máscara.
        knn_k_choices: Opciones de ``k`` cuando ``mask_mode == "knn"``.
        knn_k_default: Valor de ``k`` usado cuando ``mask_mode`` no requiere
            muestreo KNN.
        allow_self: Si es ``True``, permite auto-conexiones.
        sym: Si es ``True``, fuerza simetría.
        compute_dist_matrix: Si es ``True``, calcula matriz de distancias.
            Aunque sea ``False`` en el JSON, ``hpo/sampling.py`` debe forzarlo
            a ``True`` cuando ``edge_feature_mode == "distance_v1"``.
        apply_mask_to_loss_choices: Opciones para aplicar máscara a la loss.
        sinkhorn_nll_weight_choices: Pesos posibles para la pérdida auxiliar
            NLL sobre ``transition_probs``. ``0.0`` conserva el comportamiento
            antiguo; valores como ``0.05`` o ``0.1`` activan supervisión auxiliar.
        use_separate_loss_mask_choices: Metadata/restricción para documentar que
            el estudio usa separación de máscaras. La separación ya es fija en
            el código y no se pasa a ``TrainingParams``.
        device: Device fijo del entrenamiento.
        verbose: Si es ``True``, habilita logging detallado del trainer.
        patience: Patience del early stopping principal.
        min_delta: Mejora mínima para considerar avance en validación.
        overfit_patience: Patience para stop por sobreajuste.
        overfit_min_delta: Umbral mínimo para detectar sobreajuste.
    """

    # Repro / split
    seed: int = 123
    val_frac: float = 0.20

    # DataLoader
    batch_size_choices: Sequence[int] = (8, 16, 32)
    shuffle_train: bool = True
    num_workers: int = 0

    # Optimización
    epochs: int = 200
    lr_low: float = 1e-5
    lr_high: float = 1e-3
    lr_log: bool = True
    weight_decay_low: float = 1e-6
    weight_decay_high: float = 1e-2
    weight_decay_log: bool = True
    grad_clip_norm_choices: Sequence[float] = (0.5, 1.0, 2.0)
    use_amp_choices: Sequence[Optional[bool]] = (None,)

    # Tour + máscaras
    mask_mode_choices: Sequence[str] = ("dense", "knn")
    knn_k_choices: Sequence[int] = (5, 10)
    knn_k_default: int = 10
    allow_self: bool = False
    sym: bool = True
    compute_dist_matrix: bool = True
    apply_mask_to_loss_choices: Sequence[bool] = (True,)

    # Pérdida auxiliar y separación de máscaras.
    # sinkhorn_nll_weight=0.0 mantiene comportamiento antiguo.
    # Valores > 0 activan supervisión auxiliar sobre transition_probs.
    sinkhorn_nll_weight_choices: Sequence[float] = (0.05, 0.1)

    # Metadata/restricción: la separación sinkhorn_mask/loss_mask/decoder_mask
    # ya está implementada de forma fija; no se pasa a TrainingParams.
    use_separate_loss_mask_choices: Sequence[bool] = (True,)

    # Device
    device: Optional[str] = None

    # Logging / debug / early stopping
    verbose: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    overfit_patience: Optional[int] = 5
    overfit_min_delta: float = 1e-4


@dataclass
class OptunaSearchConfig:
    """Agrupa la configuración completa de un estudio.

    Attributes:
        workflow: Configuración operativa del estudio y de su storage.
        model_space: Search space del modelo.
        train_space: Search space del entrenamiento.
    """

    workflow: OptunaWorkflowParams = field(default_factory=OptunaWorkflowParams)
    model_space: OptunaModelSearchSpace = field(default_factory=OptunaModelSearchSpace)
    train_space: OptunaTrainSearchSpace = field(default_factory=OptunaTrainSearchSpace)


# ============================================================================
# Defaults del módulo
# ============================================================================
# Estos defaults solo sirven para simplificar firmas de funciones y scripts.
# No se usan para reconstruir el mejor trial desde la DB.
DEFAULT_OPTUNA_WORKFLOW_PARAMS = OptunaWorkflowParams()
DEFAULT_OPTUNA_MODEL_SPACE = OptunaModelSearchSpace()
DEFAULT_OPTUNA_TRAIN_SPACE = OptunaTrainSearchSpace()
DEFAULT_OPTUNA_SEARCH_CONFIG = OptunaSearchConfig()


# ============================================================================
# Helpers de validación de configuración
# ============================================================================


def _validate_workflow_cfg(workflow_cfg: OptunaWorkflowParams) -> None:
    """Valida combinaciones de configuración del workflow de Optuna.

    Args:
        workflow_cfg: Configuración operativa del estudio.

    Raises:
        ValueError: Si la configuración tiene combinaciones inválidas.
    """
    if workflow_cfg.direction not in ("minimize", "maximize"):
        raise ValueError(
            "OptunaWorkflowParams.direction debe ser 'minimize' o 'maximize'. "
            f"Valor recibido: {workflow_cfg.direction!r}"
        )

    timeout_values = {
        "timeout_days": workflow_cfg.timeout_days,
        "timeout_hours": workflow_cfg.timeout_hours,
        "timeout_minutes": workflow_cfg.timeout_minutes,
        "timeout_seconds": workflow_cfg.timeout_seconds,
    }

    for name, value in timeout_values.items():
        if value is not None and float(value) < 0.0:
            raise ValueError(
                f"OptunaWorkflowParams.{name} no puede ser negativo. "
                f"Valor recibido: {value!r}"
            )

    if workflow_cfg.export_best_from_current_run and workflow_cfg.retrain_best_at_end:
        raise ValueError(
            "Configuración inválida: export_best_from_current_run=True y "
            "retrain_best_at_end=True no deben usarse al mismo tiempo."
        )

    if workflow_cfg.export_best_from_current_run and workflow_cfg.n_jobs != 1:
        raise ValueError(
            "Configuración inválida: export_best_from_current_run=True requiere "
            "n_jobs=1 para evitar condiciones de carrera al guardar el mejor "
            "candidato de la corrida."
        )