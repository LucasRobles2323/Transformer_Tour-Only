#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/plot_io.py

"""Exportación de gráficos de entrenamiento.

Este módulo lee historiales de entrenamiento almacenados en JSON y genera
gráficos PNG para métricas como pérdida y precisión. Usa matplotlib con backend
``Agg`` para poder ejecutarse en entornos sin interfaz gráfica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .keys import (
    DEFAULT_DPI,
    KEY_EPOCH,
    KEY_HISTORY,
    KEY_HISTORY_FILE,
    KEY_RUN_TAG,
    KEY_TRAIN_ACC,
    KEY_TRAIN_LOSS,
    KEY_VAL_ACC,
    KEY_VAL_LOSS,
)
from .json_io import load_json
from .runs_io import read_runs_index, select_run_record
from .paths import (
    build_history_dir_path,
    build_plot_acc_path,
    build_plot_loss_path,
)


def _get_pyplot():
    """Importa matplotlib bajo demanda usando backend no interactivo.

    Returns:
        Módulo ``matplotlib.pyplot``.

    Notes:
        Se usa import local porque matplotlib es una dependencia relativamente
        pesada y solo se necesita al exportar gráficos.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _as_float_list(values: Sequence[Any]) -> List[float]:
    """Convierte una secuencia de valores a lista de floats.

    Si un valor no puede convertirse a ``float``, se reemplaza por ``nan`` para
    mantener la longitud de la serie.

    Args:
        values: Secuencia de valores numéricos o convertibles a número.

    Returns:
        Lista de valores ``float``.
    """
    out: List[float] = []

    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(float("nan"))

    return out


def _extract_series(
    history: List[Dict[str, Any]],
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Extrae series de métricas desde un historial de entrenamiento.

    Args:
        history: Lista de registros por época.

    Returns:
        Tupla ``(epochs, train_loss, val_loss, train_acc, val_acc)``.
    """
    if not history:
        return [], [], [], [], []

    epochs: List[int] = []
    train_loss: List[Any] = []
    val_loss: List[Any] = []
    train_acc: List[Any] = []
    val_acc: List[Any] = []

    for index, row in enumerate(history):
        epoch = row.get(KEY_EPOCH, index + 1)

        try:
            epochs.append(int(epoch))
        except (TypeError, ValueError):
            epochs.append(index + 1)

        train_loss.append(row.get(KEY_TRAIN_LOSS, float("nan")))
        val_loss.append(row.get(KEY_VAL_LOSS, float("nan")))
        train_acc.append(row.get(KEY_TRAIN_ACC, float("nan")))
        val_acc.append(row.get(KEY_VAL_ACC, float("nan")))

    return (
        epochs,
        _as_float_list(train_loss),
        _as_float_list(val_loss),
        _as_float_list(train_acc),
        _as_float_list(val_acc),
    )


def _unique_path(path: Path) -> Path:
    """Genera una ruta disponible sin sobrescribir archivos existentes.

    Args:
        path: Ruta deseada.

    Returns:
        Ruta original si no existe; de lo contrario, una ruta versionada con
        sufijo ``_v2``, ``_v3``, etc.
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    version = 2

    while True:
        candidate = parent / f"{stem}_v{version}{suffix}"
        if not candidate.exists():
            return candidate

        version += 1


def _plot_two_lines(
    *,
    x: Sequence[int],
    y1: Sequence[float],
    y2: Sequence[float],
    title: str,
    ylabel: str,
    out_path: Path,
    label1: str,
    label2: str,
    dpi: int,
) -> Path:
    """Dibuja y guarda un gráfico de dos series.

    Args:
        x: Valores del eje X.
        y1: Primera serie.
        y2: Segunda serie.
        title: Título del gráfico.
        ylabel: Etiqueta del eje Y.
        out_path: Ruta de salida deseada.
        label1: Leyenda de la primera serie.
        label2: Leyenda de la segunda serie.
        dpi: Resolución del PNG.

    Returns:
        Ruta final del archivo guardado.
    """
    plt = _get_pyplot()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = _unique_path(out_path)

    fig, ax = plt.subplots()
    ax.plot(x, y1, label=label1)
    ax.plot(x, y2, label=label2)
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=int(dpi))
    plt.close(fig)

    return out_path


def _safe_run_tag(run_tag: Any, run_id: Optional[Union[int, str]]) -> str:
    """Construye una etiqueta de run segura.

    Args:
        run_tag: Etiqueta leída desde metadata.
        run_id: Identificador alternativo de la run.

    Returns:
        Etiqueta de run normalizada.
    """
    tag = str(run_tag or "").strip()

    if tag:
        return tag

    if run_id is None:
        return "run??"

    run_id_str = str(run_id).strip()
    return run_id_str if run_id_str.lower().startswith("run") else f"run{run_id_str}"


def export_training_plots(
    *,
    model_id: str,
    run_id: Optional[Union[int, str]] = None,
    dpi: int = DEFAULT_DPI,
) -> Dict[str, Union[Path, str]]:
    """Exporta gráficos PNG del historial de entrenamiento de un modelo.

    Lee el historial asociado a una run de entrenamiento y genera dos gráficos:
    pérdida de entrenamiento/validación y precisión de entrenamiento/validación.

    Args:
        model_id: Identificador del modelo.
        run_id: Identificador opcional de la run. Si es ``None``, se selecciona
            la run más reciente disponible.
        dpi: Resolución de salida de los archivos PNG.

    Returns:
        Diccionario con las rutas de los gráficos generados y metadatos básicos.

    Raises:
        FileNotFoundError: Si no existe el historial asociado.
        ValueError: Si el historial existe pero está vacío o tiene formato inválido.
    """
    model_id_norm, _params_root, runs = read_runs_index(model_id=model_id)
    record = select_run_record(runs, run_id)

    run_tag = _safe_run_tag(record.get(KEY_RUN_TAG, ""), run_id)
    history_file = record.get(KEY_HISTORY_FILE, None) or f"{run_tag}.history.json"

    history_root = build_history_dir_path(model_id_norm)
    history_path = (history_root / str(history_file)).resolve()

    if not history_path.exists():
        raise FileNotFoundError(f"No existe history: {history_path}")

    blob = load_json(history_path)
    history = blob.get(KEY_HISTORY, []) if isinstance(blob, dict) else []

    if not isinstance(history, list) or not history:
        raise ValueError(f"History vacío o inválido en: {history_path}")

    epochs, train_loss, val_loss, train_acc, val_acc = _extract_series(history)

    title_loss = f"Pérdida por época de {model_id_norm} en {run_tag}"
    title_acc = f"Precisión por época de {model_id_norm} en {run_tag}"

    loss_path = _plot_two_lines(
        x=epochs,
        y1=train_loss,
        y2=val_loss,
        title=title_loss,
        ylabel="Pérdida",
        out_path=build_plot_loss_path(model_id_norm, run_tag),
        label1="Entrenamiento",
        label2="Validación",
        dpi=dpi,
    )

    acc_path = _plot_two_lines(
        x=epochs,
        y1=train_acc,
        y2=val_acc,
        title=title_acc,
        ylabel="Precisión",
        out_path=build_plot_acc_path(model_id_norm, run_tag),
        label1="Entrenamiento",
        label2="Validación",
        dpi=dpi,
    )

    return {
        "loss": loss_path,
        "accuracy": acc_path,
        "model_id": model_id_norm,
        "run_tag": run_tag,
    }