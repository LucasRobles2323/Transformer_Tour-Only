#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/hpo/callbacks.py

"""Callbacks y control operativo de estudios Optuna.

Este módulo concentra el manejo de interrupción por teclado, conteo de trials
de la corrida actual y detención suave por timeout.
"""

from __future__ import annotations

import time
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, Optional

from src.ttp_packages.infrastructure.logging import setup_logger

if TYPE_CHECKING:
    import optuna

logger = setup_logger(__name__)

# ============================================================================
# Control de interrupción por teclado
# ============================================================================

# Bandera global para pedir parada suave: termina el trial actual y luego detiene
# el estudio.
STOP_REQUESTED = False


def reset_stop_requested() -> None:
    """Reinicia la bandera global de parada suave."""
    global STOP_REQUESTED
    STOP_REQUESTED = False


def is_stop_requested() -> bool:
    """Indica si se solicitó parada suave por Ctrl+C.

    Returns:
        ``True`` si hay una parada suave pendiente; ``False`` en caso contrario.
    """
    return STOP_REQUESTED


def _handle_sigint(signum: int, frame: FrameType | None) -> None:
    """Maneja Ctrl+C con parada suave en el primer intento.

    Args:
        signum: Señal recibida por el proceso.
        frame: Frame actual de ejecución asociado a la señal.

    Raises:
        KeyboardInterrupt: Si el usuario presiona Ctrl+C por segunda vez.
    """
    global STOP_REQUESTED

    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n[STOP] Ctrl+C recibido. Se detendrá después del trial actual.\n")
    else:
        print("\n[STOP] Segundo Ctrl+C. Abortando inmediatamente.")
        raise KeyboardInterrupt


def _stop_after_current_trial_callback(
    study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Detiene el estudio una vez que termina el trial actual.

    Esta callback se ejecuta al final de cada trial. Si la bandera global
    ``STOP_REQUESTED`` está activa, llama a ``study.stop()`` para impedir que se
    lancen nuevos trials.

    Args:
        study: Estudio activo de Optuna.
        trial: Trial que acaba de finalizar.
        log_fn: Función opcional de logging.
    """
    if not STOP_REQUESTED:
        return

    if log_fn is None:
        log_fn = logger.info

    log_fn(
        "[OPTUNA_WORK] Stop solicitado. "
        f"Trial actual completado: #{trial.number}. "
        "No se lanzarán más trials.\n"
    )

    study.stop()


# ============================================================================
# Callbacks de control del estudio
# ============================================================================


def _count_current_run_trial_callback(
    study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
    tracker: dict[str, Any],
) -> None:
    """Cuenta trials finalizados durante la corrida actual.

    Args:
        study: Estudio activo de Optuna.
        trial: Trial que acaba de finalizar.
        tracker: Tracker mutable de la corrida actual.
    """
    # Import local: TrialState solo se necesita al clasificar el estado del trial.
    from optuna.trial import TrialState

    tracker["finished_trials_this_run"] = (
        int(tracker.get("finished_trials_this_run", 0)) + 1
    )

    if trial.state == TrialState.COMPLETE:
        tracker["completed_trials_this_run"] = (
            int(tracker.get("completed_trials_this_run", 0)) + 1
        )


def _stop_after_timeout_callback(
    study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
    tracker: dict[str, Any],
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Detiene el estudio después del trial actual si se alcanzó el timeout.

    Esta callback no interrumpe el trial en ejecución. Solo impide que se lancen
    nuevos trials después de que el actual termina.

    Args:
        study: Estudio activo de Optuna.
        trial: Trial que acaba de finalizar.
        tracker: Tracker mutable de la corrida actual. Debe contener
            ``started_at`` y puede contener ``timeout_seconds``.
        log_fn: Función opcional de logging.
    """
    timeout_seconds = tracker.get("timeout_seconds")
    if timeout_seconds is None:
        return

    elapsed_seconds = time.monotonic() - float(tracker["started_at"])

    if elapsed_seconds < float(timeout_seconds):
        return

    tracker["stop_reason"] = "time_limit_reached"

    if log_fn is None:
        log_fn = logger.info

    log_fn(
        "[OPTUNA_WORK] Time limit alcanzado. "
        f"Trial actual completado: #{trial.number}. "
        "No se lanzarán más trials.\n"
    )

    study.stop()