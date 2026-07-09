#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/hpo/study.py

"""Creación, inspección y mantenimiento de estudios Optuna.

Este módulo concentra el manejo del storage, creación/carga de estudios,
lectura estricta del mejor trial y limpieza de trials ``RUNNING``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Callable, Optional

import optuna
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from optuna.trial import TrialState

from src.ttp_packages.hpo.config import (
    DEFAULT_OPTUNA_SEARCH_CONFIG,
    OptunaSearchConfig,
    OptunaWorkflowParams,
)
from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.modeling.config import TTPArchitectureParams
from src.ttp_packages.training.config import TrainingParams
from src.ttp_packages.infrastructure.log_format import log_key_value_block


logger = setup_logger(__name__)


# ============================================================================
# Helpers de storage y ciclo de vida del estudio
# ============================================================================


def _build_storage(
    workflow_cfg: OptunaWorkflowParams,
) -> RDBStorage:
    """Construye el storage persistente de Optuna.

    Args:
        workflow_cfg: Configuración operativa del estudio.

    Returns:
        Storage relacional configurado para Optuna.
    """
    return RDBStorage(
        url=workflow_cfg.storage_url,
        heartbeat_interval=workflow_cfg.heartbeat_interval,
        grace_period=workflow_cfg.grace_period,
    )


def create_optuna_study(
    search_cfg: OptunaSearchConfig = DEFAULT_OPTUNA_SEARCH_CONFIG,
) -> optuna.study.Study:
    """Crea o carga un estudio Optuna y guarda snapshots de configuración.

    Args:
        search_cfg: Configuración completa del estudio.

    Returns:
        Estudio de Optuna listo para usar.
    """
    workflow_cfg = search_cfg.workflow

    storage = _build_storage(workflow_cfg)
    sampler = TPESampler(seed=workflow_cfg.sampler_seed)

    study = optuna.create_study(
        study_name=workflow_cfg.study_name,
        storage=storage,
        sampler=sampler,
        direction=workflow_cfg.direction,
        load_if_exists=workflow_cfg.load_if_exists,
    )

    # Guarda snapshots para auditar el workflow y search space del estudio.
    if "workflow_cfg" not in study.user_attrs:
        study.set_user_attr("workflow_cfg", asdict(search_cfg.workflow))
    if "model_space_cfg" not in study.user_attrs:
        study.set_user_attr("model_space_cfg", asdict(search_cfg.model_space))
    if "train_space_cfg" not in study.user_attrs:
        study.set_user_attr("train_space_cfg", asdict(search_cfg.train_space))

    return study


# ============================================================================
# Helpers de inspección y lectura estricta del estudio
# ============================================================================


def inspect_optuna_tour_study(
    search_cfg: OptunaSearchConfig = DEFAULT_OPTUNA_SEARCH_CONFIG,
    *,
    print_result: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Inspecciona el estado actual del estudio y resume su mejor trial.

    Args:
        search_cfg: Configuración completa del estudio.
        print_result: Si es ``True``, registra el resumen por ``log_fn``.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Diccionario con conteos por estado, trials ``RUNNING`` e información del
        mejor trial si existe.
    """
    if log_fn is None:
        log_fn = logger.info

    study = create_optuna_study(search_cfg=search_cfg)
    trials = study.get_trials(deepcopy=False)

    counts = {
        "n_trials": len(trials),
        "n_complete": sum(t.state == TrialState.COMPLETE for t in trials),
        "n_running": sum(t.state == TrialState.RUNNING for t in trials),
        "n_fail": sum(t.state == TrialState.FAIL for t in trials),
        "n_pruned": sum(t.state == TrialState.PRUNED for t in trials),
        "n_waiting": sum(t.state == TrialState.WAITING for t in trials),
    }

    running_trials = [
        {
            "number": trial.number,
            "trial_id": trial._trial_id,
            "state": trial.state.name,
            "datetime_start": (
                None
                if trial.datetime_start is None
                else trial.datetime_start.isoformat()
            ),
            "params": dict(trial.params),
            "user_attrs": dict(trial.user_attrs),
        }
        for trial in trials
        if trial.state == TrialState.RUNNING
    ]

    best_info = None
    try:
        best_info = get_best_tour_trial_result(
            search_cfg=search_cfg,
            print_result=False,
            log_fn=log_fn,
        )
    except RuntimeError:
        # Un estudio sin trials COMPLETE todavía no tiene best_trial disponible.
        best_info = None

    result = {
        "study_name": search_cfg.workflow.study_name,
        "counts": counts,
        "running_trials": running_trials,
        "best_trial_number": (
            None if best_info is None else best_info["best_trial_number"]
        ),
        "best_value": (
            None if best_info is None else best_info["best_value"]
        ),
        "best_params": (
            None if best_info is None else best_info["best_params"]
        ),
        "final_model_params": (
            None
            if best_info is None
            else asdict(best_info["final_model_params"])
        ),
        "final_train_params": (
            None
            if best_info is None
            else asdict(best_info["final_train_params"])
        ),
        "best_trial_user_attrs": (
            None if best_info is None else best_info["user_attrs"]
        ),
    }

    if print_result:
        log_fn("\n[OPTUNA] Resumen del estudio")
        log_fn(f"  study_name: {result['study_name']}")
        log_fn(f"  counts: {result['counts']}")

        if best_info is None:
            log_fn("  best_trial: no hay trials COMPLETE todavía.")
        else:
            log_fn(f"  best_trial_number: {result['best_trial_number']}")
            log_fn(f"  best_value: {result['best_value']}")
            log_fn("  best_params:")
            log_fn(json.dumps(result["best_params"], indent=2, ensure_ascii=False))
            log_fn("  final_model_params:")
            log_fn(
                json.dumps(
                    result["final_model_params"],
                    indent=2,
                    ensure_ascii=False,
                )
            )
            log_fn("  final_train_params:")
            log_fn(
                json.dumps(
                    result["final_train_params"],
                    indent=2,
                    ensure_ascii=False,
                )
            )

        if running_trials:
            log_fn("  running_trials:")
            for row in running_trials:
                log_fn(
                    json.dumps(
                        {
                            "number": row["number"],
                            "trial_id": row["trial_id"],
                            "datetime_start": row["datetime_start"],
                        },
                        ensure_ascii=False,
                    )
                )
        else:
            log_fn("  running_trials: []")

    return result


def get_best_tour_trial_result(
    search_cfg: OptunaSearchConfig = DEFAULT_OPTUNA_SEARCH_CONFIG,
    *,
    print_result: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Recupera el mejor trial del estudio de forma estricta.

    La función no reconstruye parámetros desde defaults. Si el mejor trial no
    guardó ``final_model_params`` o ``final_train_params`` en ``user_attrs``,
    considera que la información es insuficiente.

    Args:
        search_cfg: Configuración completa del estudio.
        print_result: Si es ``True``, registra el resultado por ``log_fn``.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Diccionario con nombre del estudio, número del mejor trial, valor,
        parámetros de Optuna, parámetros finales reconstruidos y ``user_attrs``.

    Raises:
        RuntimeError: Si el estudio no tiene trials completados o si el mejor
            trial no guardó snapshots completos en ``user_attrs``.
    """
    if log_fn is None:
        log_fn = logger.info

    study = create_optuna_study(search_cfg=search_cfg)

    try:
        best_trial = study.best_trial
    except ValueError as error:
        raise RuntimeError("El estudio no tiene trials completados todavía.") from error

    if "final_model_params" not in best_trial.user_attrs:
        raise RuntimeError(
            "El best_trial no tiene 'final_model_params' guardado en user_attrs."
        )

    if "final_train_params" not in best_trial.user_attrs:
        raise RuntimeError(
            "El best_trial no tiene 'final_train_params' guardado en user_attrs."
        )

    final_model_params = TTPArchitectureParams(
        **best_trial.user_attrs["final_model_params"]
    )
    final_train_params = TrainingParams(
        **best_trial.user_attrs["final_train_params"]
    )

    result = {
        "study_name": search_cfg.workflow.study_name,
        "best_trial_number": best_trial.number,
        "best_value": study.best_value,
        "best_params": dict(study.best_params),
        "final_model_params": final_model_params,
        "final_train_params": final_train_params,
        "user_attrs": dict(best_trial.user_attrs),
    }

    if print_result:
        log_key_value_block(
            log_fn,
            "\n[OPTUNA] Mejor trial encontrado",
            {
                "study_name": result["study_name"],
                "best_trial_number": result["best_trial_number"],
                "best_value": result["best_value"],
                "best_params": result["best_params"],
                "final_model_params": asdict(result["final_model_params"]),
                "final_train_params": asdict(result["final_train_params"]),
            },
            key_order=[
                "study_name",
                "best_trial_number",
                "best_value",
                "best_params",
                "final_model_params",
                "final_train_params",
            ],
            indent=2,
            nested_indent=4,
        )
    
    return result


def fail_running_trials(
    search_cfg: OptunaSearchConfig = DEFAULT_OPTUNA_SEARCH_CONFIG,
    *,
    only_stale: bool = False,
    print_result: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Marca trials ``RUNNING`` como ``FAIL``.

    Args:
        search_cfg: Configuración del estudio.
        only_stale: Si es ``True``, solo intenta fallar trials stale usando la
            utilidad de Optuna. Si es ``False``, fuerza todos los ``RUNNING`` a
            ``FAIL``.
        print_result: Si es ``True``, registra el resultado por ``log_fn``.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Diccionario con el modo de limpieza y el detalle de los trials afectados.
    """
    if log_fn is None:
        log_fn = logger.info

    study = create_optuna_study(search_cfg=search_cfg)
    storage = study._storage

    failed_numbers: list[int] = []

    if only_stale:
        optuna.storages.fail_stale_trials(study)

        # Relee los trials porque fail_stale_trials puede cambiar estados en storage.
        trials_after = study.get_trials(deepcopy=False)
        failed_numbers = [
            trial.number
            for trial in trials_after
            if trial.state == TrialState.FAIL
        ]

        out = {
            "mode": "only_stale",
            "study_name": search_cfg.workflow.study_name,
            "message": "Se intentó marcar como FAIL solo trials stale.",
            "failed_trial_numbers": failed_numbers,
        }
    else:
        running_trials = study.get_trials(
            deepcopy=False,
            states=(TrialState.RUNNING,),
        )

        changed = []
        for trial in running_trials:
            updated = storage.set_trial_state_values(
                trial._trial_id,
                state=TrialState.FAIL,
            )
            if updated:
                changed.append(trial.number)

        out = {
            "mode": "all_running",
            "study_name": search_cfg.workflow.study_name,
            "n_running_found": len(running_trials),
            "failed_trial_numbers": changed,
        }

    if print_result:
        log_fn("\n[OPTUNA] Limpieza de RUNNING")
        log_fn(json.dumps(out, indent=2, ensure_ascii=False))

    return out