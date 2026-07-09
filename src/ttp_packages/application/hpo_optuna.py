#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/hpo_optuna.py

"""Workflows de HPO con Optuna para el modelo tour-only.

Este módulo conecta Optuna con los workflows de construcción, entrenamiento y
exportación de modelos. La capa de scripts debe importar configuraciones desde
``application.config`` y llamar a los workflows definidos aquí.
"""

from __future__ import annotations

import signal
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from .build_model import instantiate_ttp_model
from .train_model import train_tour_only_work

from src.ttp_packages.hpo.callbacks import (
    _count_current_run_trial_callback,
    _handle_sigint,
    _stop_after_current_trial_callback,
    _stop_after_timeout_callback,
    is_stop_requested,
    reset_stop_requested,
)
from src.ttp_packages.hpo.config import (
    DEFAULT_OPTUNA_SEARCH_CONFIG,
    OptunaSearchConfig,
    _validate_workflow_cfg,
)
from src.ttp_packages.hpo.results import (
    BestArtifactCandidate,
    OptunaStudyRunResult,
    _copy_state_dict_to_cpu,
    _is_better,
)
from src.ttp_packages.hpo.sampling import (
    _sample_model_params,
    _sample_train_params,
)
from src.ttp_packages.hpo.study import (
    create_optuna_study,
    fail_running_trials,
    get_best_tour_trial_result,
)
from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.infrastructure.log_format import log_key_value_block
from src.ttp_packages.infrastructure.storage.model_out import export_training_artifacts
from src.ttp_packages.ml_data.torch.dataset import TTPTensorDataset
from src.ttp_packages.training.utils import seed_everything

if TYPE_CHECKING:
    import optuna
    import torch

    from src.ttp_packages.modeling.config import TTPArchitectureParams
    from src.ttp_packages.training.config import TrainingParams


logger = setup_logger(__name__)


# ============================================================================
# Build + train por trial
# ============================================================================


def build_and_train_tour_work(
    dataset: Any,
    *,
    model_params: TTPArchitectureParams,
    train_params: TrainingParams,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[
    torch.nn.Module,
    list[Dict[str, Any]],
    Dict[str, Any],
    TTPArchitectureParams,
    TrainingParams,
]:
    """Construye el modelo y ejecuta el entrenamiento tour-only.

    Args:
        dataset: Dataset ya cargado para entrenamiento.
        model_params: Configuración completa del modelo.
        train_params: Configuración completa del entrenamiento.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Tupla ``(trained_model, history, summary, model_params, train_params)``.
    """
    if log_fn is None:
        log_fn = logger.info

    # La semilla se fija antes de instanciar el modelo para reproducir pesos
    # iniciales, split y operaciones aleatorias del entrenamiento.
    seed_everything(train_params.seed)

    model = instantiate_ttp_model(
        params=model_params,
        device=train_params.device,
        eval_mode=False,
    )

    trained_model, history, summary = train_tour_only_work(
        model=model,
        dataset=dataset,
        train_params=train_params,
        log_fn=log_fn,
    )

    if train_params.verbose:
        log_fn(
            "[OPTUNA_WORK] Entrenamiento finalizado | "
            f"best_epoch={summary.get('best_epoch')} | "
            f"best_val_loss={summary.get('best_val_loss')}\n"
        )

    return trained_model, history, summary, model_params, train_params


# ============================================================================
# Exportación de artefactos del mejor trial
# ============================================================================


def export_best_candidate_artifacts(
    candidate: BestArtifactCandidate,
    *,
    model_id: str,
    overwrite_model: bool = True,
    overwrite_history: bool = False,
    export_plots: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Exporta artefactos del mejor candidato de la corrida sin reentrenar.

    Reconstruye la arquitectura, carga los pesos entrenados del candidato y usa
    el flujo estándar de exportación. No vuelve a llamar al trainer.

    Args:
        candidate: Mejor candidato encontrado durante la corrida actual.
        model_id: Identificador con el que se guardará el modelo.
        overwrite_model: Si es ``True``, permite sobrescribir el checkpoint.
        overwrite_history: Si es ``True``, permite sobrescribir el history.
        export_plots: Si es ``True``, exporta plots del run.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Diccionario con metadatos del candidato y rutas exportadas.
    """
    if log_fn is None:
        log_fn = logger.info

    model = instantiate_ttp_model(
        params=candidate.final_model_params,
        device=candidate.final_train_params.device,
        eval_mode=False,
    )

    model.load_state_dict(candidate.model_state_dict)

    export_out = export_training_artifacts(
        model_id=model_id,
        model=model,
        model_params=candidate.final_model_params,
        train_params=candidate.final_train_params,
        summary=candidate.summary,
        history=candidate.history,
        overwrite_model=overwrite_model,
        overwrite_history=overwrite_history,
    )

    # params_run identifica la corrida exportada con el formato del proyecto.
    run_tag = export_out["params_run"].stem

    plots_out = None
    if export_plots:
        # Import local: evita cargar dependencias de plotting si no se exportan plots.
        from src.ttp_packages.infrastructure.storage.plot_io import (
            export_training_plots,
        )

        plots_out = export_training_plots(
            model_id=model_id,
            run_id=run_tag,
        )

    log_fn(
        "[OPTUNA_WORK] Candidato exportado sin reentrenar | "
        f"trial={candidate.trial_number} | "
        f"value={candidate.value} | "
        f"run_tag={run_tag}\n"
    )

    return {
        "best_trial_number": candidate.trial_number,
        "best_value": candidate.value,
        "best_params": candidate.trial_params,
        "final_model_params": candidate.final_model_params,
        "final_train_params": candidate.final_train_params,
        "summary": candidate.summary,
        "run_tag": run_tag,
        "export_paths": export_out,
        "plot_paths": plots_out,
        "retrained": False,
    }


def save_best_tour_trial_artifacts(
    dataset: Any,
    *,
    model_id: str,
    search_cfg: OptunaSearchConfig = DEFAULT_OPTUNA_SEARCH_CONFIG,
    overwrite_model: bool = True,
    overwrite_history: bool = False,
    export_plots: bool = True,
    print_result: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Reentrena y exporta artefactos del mejor trial del estudio.

    La reconstrucción del mejor trial usa únicamente snapshots completos
    guardados en ``user_attrs``. No reconstruye parámetros desde defaults.

    Args:
        dataset: Dataset ya cargado.
        model_id: Identificador con el que se guardará el modelo.
        search_cfg: Configuración del estudio.
        overwrite_model: Si es ``True``, permite sobrescribir el checkpoint.
        overwrite_history: Si es ``True``, permite sobrescribir el history.
        export_plots: Si es ``True``, exporta plots del run.
        print_result: Si es ``True``, registra el mejor trial antes de exportar.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Diccionario con metadatos del mejor trial y rutas exportadas.

    Raises:
        RuntimeError: Si el workflow tiene deshabilitado el guardado de artefactos.
    """
    if log_fn is None:
        log_fn = logger.info

    if not search_cfg.workflow.save_best_artifacts:
        raise RuntimeError(
            "save_best_artifacts=False en OptunaWorkflowParams. "
            "El guardado de artefactos del mejor trial está deshabilitado."
        )

    best_info = get_best_tour_trial_result(
        search_cfg=search_cfg,
        print_result=print_result,
        log_fn=log_fn,
    )

    trained_model, history, summary, final_model_params, final_train_params = (
        build_and_train_tour_work(
            dataset=dataset,
            model_params=best_info["final_model_params"],
            train_params=best_info["final_train_params"],
            log_fn=log_fn,
        )
    )

    export_out = export_training_artifacts(
        model_id=model_id,
        model=trained_model,
        model_params=final_model_params,
        train_params=final_train_params,
        summary=summary,
        history=history,
        overwrite_model=overwrite_model,
        overwrite_history=overwrite_history,
    )

    # params_run identifica la corrida exportada con el formato del proyecto.
    run_tag = export_out["params_run"].stem

    plots_out = None
    if export_plots:
        # Import local: evita cargar dependencias de plotting si no se exportan plots.
        from src.ttp_packages.infrastructure.storage.plot_io import (
            export_training_plots,
        )

        plots_out = export_training_plots(
            model_id=model_id,
            run_id=run_tag,
        )

    return {
        "best_trial_number": best_info["best_trial_number"],
        "best_value": best_info["best_value"],
        "best_params": best_info["best_params"],
        "final_model_params": final_model_params,
        "final_train_params": final_train_params,
        "summary": summary,
        "run_tag": run_tag,
        "export_paths": export_out,
        "plot_paths": plots_out,
    }


# ============================================================================
# Función objetivo de Optuna
# ============================================================================


def objective_tour_only(
    trial: optuna.trial.Trial,
    dataset: Any,
    search_cfg: OptunaSearchConfig = DEFAULT_OPTUNA_SEARCH_CONFIG,
    log_fn: Optional[Callable[[str], None]] = None,
    artifact_tracker: Optional[Dict[str, Any]] = None,
) -> float:
    """Función objetivo de Optuna para entrenamiento tour-only.

    Args:
        trial: Trial actual de Optuna.
        dataset: Dataset ya cargado.
        search_cfg: Configuración completa del estudio.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.
        artifact_tracker: Tracker opcional para conservar el mejor candidato
            entrenado durante esta corrida.

    Returns:
        Mejor ``val_loss`` alcanzado por el trial.
    """
    if log_fn is None:
        log_fn = logger.info

    final_model_params = _sample_model_params(
        trial=trial,
        model_space=search_cfg.model_space,
    )
    final_train_params = _sample_train_params(
        trial=trial,
        train_space=search_cfg.train_space,
    )

    trained_model, history, summary, _, _ = build_and_train_tour_work(
        dataset=dataset,
        model_params=final_model_params,
        train_params=final_train_params,
        log_fn=log_fn,
    )

    current_value = float(summary["best_val_loss"])

    # Estos snapshots son la fuente de verdad para reconstruir el mejor trial.
    trial.set_user_attr("final_model_params", asdict(final_model_params))
    trial.set_user_attr("final_train_params", asdict(final_train_params))
    trial.set_user_attr("training_summary", summary)
    trial.set_user_attr("best_epoch", summary.get("best_epoch"))
    trial.set_user_attr("best_val_loss", summary.get("best_val_loss"))
    trial.set_user_attr(
        "best_val_acc_at_best_loss",
        summary.get("best_val_acc_at_best_loss"),
    )
    trial.set_user_attr("best_val_acc_seen", summary.get("best_val_acc_seen"))
    trial.set_user_attr("stop_reason", summary.get("stop_reason"))

    # Si existe tracker, conserva en memoria el mejor modelo nuevo de esta corrida.
    if artifact_tracker is not None:
        reference_value = (
            artifact_tracker["initial_best_value"]
            if search_cfg.workflow.save_best_only_if_improved
            else None
        )

        improves_initial_reference = _is_better(
            current_value,
            reference_value,
            direction=search_cfg.workflow.direction,
            min_delta=search_cfg.workflow.artifact_min_delta,
        )

        current_candidate = artifact_tracker.get("candidate")
        improves_current_candidate = (
            current_candidate is None
            or _is_better(
                current_value,
                current_candidate.value,
                direction=search_cfg.workflow.direction,
                min_delta=search_cfg.workflow.artifact_min_delta,
            )
        )

        if improves_initial_reference and improves_current_candidate:
            artifact_tracker["candidate"] = BestArtifactCandidate(
                trial_number=trial.number,
                value=current_value,
                model_state_dict=_copy_state_dict_to_cpu(trained_model),
                history=history,
                summary=summary,
                final_model_params=final_model_params,
                final_train_params=final_train_params,
                trial_params=dict(trial.params),
            )

            log_fn(
                "[OPTUNA_WORK] Nuevo candidato a exportar | "
                f"trial={trial.number} | "
                f"value={current_value}\n"
            )

    return current_value


# ============================================================================
# Ejecución del estudio
# ============================================================================


def run_optuna_tour_study(
    dataset: Any,
    search_cfg: OptunaSearchConfig = DEFAULT_OPTUNA_SEARCH_CONFIG,
    log_fn: Optional[Callable[[str], None]] = None,
) -> OptunaStudyRunResult:
    """Ejecuta el estudio de Optuna sobre un dataset ya cargado.

    Args:
        dataset: Dataset ya cargado.
        search_cfg: Configuración completa del estudio.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Resultado extendido de la corrida, incluyendo estudio, mejor valor
        inicial, mejor candidato nuevo y motivo de término.
    """
    if log_fn is None:
        log_fn = logger.info

    reset_stop_requested()

    workflow_cfg = search_cfg.workflow
    _validate_workflow_cfg(workflow_cfg)

    study = create_optuna_study(search_cfg=search_cfg)

    # Los trials RUNNING al inicio se tratan como huérfanos de ejecuciones previas.
    if workflow_cfg.fail_running_on_start:
        fail_running_trials(
            search_cfg=search_cfg,
            only_stale=False,
            print_result=workflow_cfg.verbose,
            log_fn=log_fn,
        )

    try:
        initial_best_value = study.best_value
        initial_best_trial_number = study.best_trial.number
    except ValueError:
        initial_best_value = None
        initial_best_trial_number = None

    tracker: Dict[str, Any] = {
        "started_at": time.monotonic(),
        "timeout_seconds": workflow_cfg.effective_timeout_seconds,
        "initial_best_value": initial_best_value,
        "initial_best_trial_number": initial_best_trial_number,
        "candidate": None,
        "stop_reason": "unknown",
        "finished_trials_this_run": 0,
        "completed_trials_this_run": 0,
    }

    if workflow_cfg.verbose:
        log_fn(
            "[OPTUNA_WORK] Iniciando estudio | "
            f"study_name={workflow_cfg.study_name} | "
            f"storage={workflow_cfg.storage_url} | "
            f"n_trials={workflow_cfg.n_trials} | "
            f"timeout_seconds={workflow_cfg.effective_timeout_seconds} | "
            f"initial_best_trial={initial_best_trial_number} | "
            f"initial_best_value={initial_best_value}\n"
        )

    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        study.optimize(
            lambda trial: objective_tour_only(
                trial=trial,
                dataset=dataset,
                search_cfg=search_cfg,
                log_fn=log_fn,
                artifact_tracker=tracker,
            ),
            n_trials=workflow_cfg.n_trials,
            timeout=workflow_cfg.effective_timeout_seconds,
            n_jobs=workflow_cfg.n_jobs,
            gc_after_trial=workflow_cfg.gc_after_trial,
            callbacks=[
                lambda study_obj, trial_obj: _count_current_run_trial_callback(
                    study_obj,
                    trial_obj,
                    tracker=tracker,
                ),
                lambda study_obj, trial_obj: _stop_after_timeout_callback(
                    study_obj,
                    trial_obj,
                    tracker=tracker,
                    log_fn=log_fn,
                ),
                lambda study_obj, trial_obj: _stop_after_current_trial_callback(
                    study_obj,
                    trial_obj,
                    log_fn=log_fn,
                ),
            ],
        )
    finally:
        signal.signal(signal.SIGINT, old_handler)

    elapsed_seconds = time.monotonic() - float(tracker["started_at"])
    timeout_seconds = tracker.get("timeout_seconds")
    timeout_reached = (
        timeout_seconds is not None
        and elapsed_seconds >= float(timeout_seconds)
    )

    if is_stop_requested():
        stop_reason = "stop_requested"
    elif int(tracker["finished_trials_this_run"]) >= int(workflow_cfg.n_trials):
        stop_reason = "n_trials_completed"
    elif timeout_reached:
        stop_reason = "time_limit_reached"
    elif int(tracker["completed_trials_this_run"]) == 0:
        stop_reason = "no_complete_trials"
    else:
        stop_reason = str(tracker.get("stop_reason", "unknown"))

    tracker["stop_reason"] = stop_reason

    best_trial_number = None
    best_value = None
    try:
        best_trial_number = study.best_trial.number
        best_value = study.best_value
    except ValueError:
        # Puede ocurrir si no existe ningún trial COMPLETE.
        pass

    if workflow_cfg.verbose:
        log_fn(
            "[OPTUNA_WORK] Estudio finalizado | "
            f"stop_reason={stop_reason} | "
            f"finished_trials_this_run={tracker['finished_trials_this_run']} | "
            f"completed_trials_this_run={tracker['completed_trials_this_run']} | "
            f"best_trial={best_trial_number} | "
            f"best_value={best_value}\n"
        )

    return OptunaStudyRunResult(
        study=study,
        initial_best_value=initial_best_value,
        initial_best_trial_number=initial_best_trial_number,
        best_candidate=tracker["candidate"],
        stop_reason=stop_reason,
        completed_trials_this_run=int(tracker["completed_trials_this_run"]),
    )





# ============================================================================
# Workflow desde dataset serializado
# ============================================================================
def _log_optuna_work_final_summary(
    result: Dict[str, Any],
    *,
    log_fn: Callable[[str], None],
) -> None:
    """Registra el resumen final del workflow Optuna.

    Args:
        result: Diccionario retornado por ``run_optuna_tour_from_dataset_work``.
        log_fn: Función de logging.
    """
    artifacts = result.get("artifacts")

    summary_values = {
        "study_name": result.get("study_name"),
        "best_trial_number": result.get("best_trial_number"),
        "best_value": result.get("best_value"),
        "n_trials": result.get("n_trials"),
        "stop_reason": result.get("stop_reason"),
        "completed_trials_this_run": result.get("completed_trials_this_run"),
        "initial_best_trial_number": result.get("initial_best_trial_number"),
        "initial_best_value": result.get("initial_best_value"),
        "has_best_candidate": result.get("has_best_candidate"),
        "candidate_exported": result.get("candidate_exported"),
        "artifact_export_mode": result.get("artifact_export_mode"),
        "artifacts": (
            "no se exportaron artefactos"
            if artifacts is None
            else artifacts
        ),
    }

    log_key_value_block(
        log_fn,
        "\n[OK] Optuna finalizado.",
        summary_values,
        key_order=[
            "study_name",
            "best_trial_number",
            "best_value",
            "n_trials",
            "stop_reason",
            "completed_trials_this_run",
            "initial_best_trial_number",
            "initial_best_value",
            "has_best_candidate",
            "candidate_exported",
            "artifact_export_mode",
            "artifacts",
        ],
        indent=2,
        nested_indent=4,
    )

def run_optuna_tour_from_dataset_work(
    *,
    dataset_file: str,
    search_cfg: OptunaSearchConfig = DEFAULT_OPTUNA_SEARCH_CONFIG,
    model_id: Optional[str] = None,
    overwrite_model: bool = True,
    overwrite_history: bool = False,
    export_best_plots: bool = True,
    print_best_result: bool = True,
    print_final_summary: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Ejecuta Optuna desde un dataset serializado.

    Args:
        dataset_file: Ruta al dataset serializado.
        search_cfg: Configuración completa del estudio.
        model_id: Identificador opcional para exportar artefactos del mejor trial.
        overwrite_model: Si es ``True``, permite sobrescribir el checkpoint.
        overwrite_history: Si es ``True``, permite sobrescribir el history.
        export_best_plots: Si es ``True``, exporta plots del mejor trial.
        print_best_result: Si es ``True``, registra el mejor trial al finalizar.
        print_final_summary: Si es ``True``, registra el resumen final del
            workflow desde la capa de aplicación.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.

    Returns:
        Diccionario con resumen del estudio y artefactos del mejor trial, si
        corresponde.
    """
    if log_fn is None:
        log_fn = logger.info

    dataset = TTPTensorDataset.from_file(
        dataset_file,
        verbose=False,
        map_location="cpu",
        log_fn=log_fn,
    )

    run_result = run_optuna_tour_study(
        dataset=dataset,
        search_cfg=search_cfg,
        log_fn=log_fn,
    )

    study = run_result.study

    try:
        best_info = get_best_tour_trial_result(
            search_cfg=search_cfg,
            print_result=print_best_result,
            log_fn=log_fn,
        )
    except RuntimeError as error:
        best_info = None
        log_fn(
            "[OPTUNA_WORK] No hay best trial disponible después del estudio. "
            f"{error}\n"
        )

    artifacts_out = None
    candidate_exported = False

    if not search_cfg.workflow.save_best_artifacts:
        log_fn(
            "[OPTUNA_WORK] save_best_artifacts=False. "
            "No se exportan artefactos.\n"
        )

    elif model_id is None:
        log_fn(
            "[OPTUNA_WORK] model_id=None. "
            "No se exportan artefactos.\n"
        )

    elif search_cfg.workflow.export_best_from_current_run:
        if run_result.best_candidate is not None:
            artifacts_out = export_best_candidate_artifacts(
                candidate=run_result.best_candidate,
                model_id=model_id,
                overwrite_model=overwrite_model,
                overwrite_history=overwrite_history,
                export_plots=export_best_plots,
                log_fn=log_fn,
            )
            candidate_exported = True
        else:
            if run_result.initial_best_value is None:
                log_fn(
                    "[OPTUNA_WORK] No se encontró candidato exportable. "
                    "No hubo trials completos válidos en esta corrida.\n"
                )
            else:
                log_fn(
                    "[OPTUNA_WORK] No se encontró un trial mejor que el "
                    "best inicial de la DB. No se exportan artefactos.\n"
                )

    elif search_cfg.workflow.retrain_best_at_end:
        artifacts_out = save_best_tour_trial_artifacts(
            dataset=dataset,
            model_id=model_id,
            search_cfg=search_cfg,
            overwrite_model=overwrite_model,
            overwrite_history=overwrite_history,
            export_plots=export_best_plots,
            print_result=False,
            log_fn=log_fn,
        )

    result = {
        "study_name": search_cfg.workflow.study_name,
        "best_trial_number": (
            None if best_info is None else best_info["best_trial_number"]
        ),
        "best_value": (
            None if best_info is None else best_info["best_value"]
        ),
        "best_params": (
            None if best_info is None else best_info["best_params"]
        ),
        "best_info": best_info,
        "artifacts": artifacts_out,
        "n_trials": len(study.trials),
        "initial_best_value": run_result.initial_best_value,
        "initial_best_trial_number": run_result.initial_best_trial_number,
        "stop_reason": run_result.stop_reason,
        "completed_trials_this_run": run_result.completed_trials_this_run,
        "has_best_candidate": run_result.best_candidate is not None,
        "candidate_exported": candidate_exported,
        "artifact_export_mode": (
            "current_run_candidate"
            if candidate_exported
            else (
                "legacy_retrain"
                if (
                    artifacts_out is not None
                    and search_cfg.workflow.retrain_best_at_end
                )
                else None
            )
        ),
    }

    if print_final_summary:
        _log_optuna_work_final_summary(
            result,
            log_fn=log_fn,
        )

    return result