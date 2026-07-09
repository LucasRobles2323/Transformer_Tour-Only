#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/hpo/results.py

"""Estructuras auxiliares para resultados y tracking de corridas HPO.

Este módulo define contenedores para candidatos de artefactos, resultados
extendidos de estudios Optuna y helpers pequeños para comparar valores objetivo
o copiar pesos del modelo a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import optuna
    import torch

    from src.ttp_packages.modeling.config import TTPArchitectureParams
    from src.ttp_packages.training.config import TrainingParams


@dataclass
class BestArtifactCandidate:
    """Candidato a mejor artefacto encontrado durante la corrida actual.

    Esta estructura guarda los pesos del modelo como ``state_dict`` en CPU para
    poder exportarlos al final de la corrida sin retener el modelo completo.

    Attributes:
        trial_number: Número del trial de Optuna.
        value: Valor objetivo del trial.
        model_state_dict: Pesos del modelo copiados a CPU.
        history: Historial de entrenamiento del trial.
        summary: Resumen de entrenamiento del trial.
        final_model_params: Parámetros finales de arquitectura.
        final_train_params: Parámetros finales de entrenamiento.
        trial_params: Parámetros muestreados por Optuna.
    """

    trial_number: int
    value: float
    model_state_dict: dict[str, Any]
    history: list[dict[str, Any]]
    summary: dict[str, Any]
    final_model_params: TTPArchitectureParams
    final_train_params: TrainingParams
    trial_params: dict[str, Any]


@dataclass
class OptunaStudyRunResult:
    """Resultado extendido de una corrida de Optuna.

    Attributes:
        study: Estudio de Optuna ya ejecutado.
        initial_best_value: Mejor valor existente en la base de datos antes de
            esta corrida.
        initial_best_trial_number: Número del mejor trial existente antes de
            esta corrida.
        best_candidate: Mejor candidato nuevo encontrado durante esta corrida.
        stop_reason: Motivo de término del estudio.
        completed_trials_this_run: Cantidad de trials completados en esta corrida.
    """

    study: optuna.study.Study
    initial_best_value: Optional[float]
    initial_best_trial_number: Optional[int]
    best_candidate: Optional[BestArtifactCandidate]
    stop_reason: str
    completed_trials_this_run: int


def _is_better(
    value: float,
    reference: Optional[float],
    *,
    direction: str,
    min_delta: float = 0.0,
) -> bool:
    """Determina si un valor mejora una referencia según la dirección.

    Args:
        value: Valor candidato.
        reference: Valor de referencia. Si es ``None``, cualquier valor válido
            se considera mejora.
        direction: Dirección de optimización. Debe ser ``"minimize"`` o
            ``"maximize"``.
        min_delta: Margen mínimo requerido para considerar mejora.

    Returns:
        ``True`` si ``value`` mejora a ``reference``.

    Raises:
        ValueError: Si ``direction`` no es soportada.
    """
    if reference is None:
        return True

    if direction == "minimize":
        return value < reference - min_delta

    if direction == "maximize":
        return value > reference + min_delta

    raise ValueError(f"direction no soportada: {direction!r}")


def _copy_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, Any]:
    """Copia los pesos del modelo a CPU.

    Esto evita retener el modelo completo en GPU mientras siguen corriendo otros
    trials.

    Args:
        model: Modelo entrenado.

    Returns:
        ``state_dict`` con tensores desconectados, movidos a CPU y clonados.
    """
    return {
        # detach evita guardar grafo; cpu libera dependencia del device; clone
        # crea una copia estable para exportar el artefacto al final.
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }