#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/compare_io.py

"""Persistencia de resultados de comparación y evaluación.

Este módulo guarda corridas de comparación en archivos JSON dentro del
directorio configurado para resultados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .file_names import ensure_json_suffix, ensure_png_suffix
from .json_io import dump_json
from .paths import build_compare_results_path

DEFAULT_COMPARE_TABLE_EXCLUDE_COLUMNS = (
    "seed",
    "pred_tour_issues",
    "tsp_tour_issues",
    "cs2sar_tour_issues",
    "random_tour_issues",
    "rand_krp_tour_issues",
)

DEFAULT_COMPARE_TABLE_HEADER_ORDER = (
    # Identificación de la instancia evaluada.
    "instance_id",

    # Valor función objetivo del modelo separado por fuente del decoder.
    "model_log_obj",    # Tour modelo desde logits + KRP.
    "model_prob_obj",   # Tour modelo desde probabilidades + KRP.

    # Baselines.
    "random_obj",       # Tour random + packing random.
    "rand_krp_obj",     # Tour random + packing KRP.
    "tsp_obj",          # Tour TSP + KRP.
    "cs2sar_obj",       # Solución CS2SA-R.

    # Métricas estructurales del modelo separadas por fuente.
    "edge_overlap_cs2sar_log",
    "edge_overlap_cs2sar_prob",
    "tsp_distance_penalty_log",
    "tsp_distance_penalty_prob",

    # Tiempo de viaje TTP, no tiempo de cómputo.
    "model_log_time",
    "model_prob_time",
    "random_time",
    "rand_krp_time",
    "tsp_time",
    "cs2sar_time",

    # Profit de la solución.
    # "model_log_profit",
    # "model_prob_profit",
    # "random_profit",
    # "rand_krp_profit",
    # "tsp_profit",
    # "cs2sar_profit",

    # Validez estructural de los tours.
    "all_tours_valid",
)

def _is_simple_table_value(value: Any) -> bool:
    """Indica si un valor puede mostrarse directamente en una celda de tabla.

    Args:
        value: Valor a evaluar.

    Returns:
        ``True`` si el valor es simple; ``False`` si es una estructura anidada.
    """
    return isinstance(value, (str, int, float, bool, type(None)))


def _format_table_value(value: Any) -> str:
    """Convierte un valor a texto para mostrarlo en una tabla.

    Los floats se formatean con máximo dos decimales.

    Args:
        value: Valor a convertir.

    Returns:
        Representación en texto apta para tabla.
    """
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")

    if value is None:
        return ""

    return str(value)


def _save_results_table_image(
    *,
    results: Sequence[Mapping[str, Any]],
    out_path: Path,
    title: str = "Resultados Comparación",
    max_rows: Optional[int] = 50,
    exclude_columns: Optional[Sequence[str]] = None,
    header_order: Optional[Sequence[str]] = None,
) -> Path:
    """Guarda una tabla de resultados como imagen PNG.

    Args:
        results: Lista de diccionarios con los resultados.
        out_path: Ruta de salida del archivo PNG.
        title: Título mostrado sobre la tabla.
        max_rows: Máximo de filas a renderizar. Si es ``None``, usa todas.
        exclude_columns: Columnas que no deben mostrarse en la tabla.
        header_order: Orden preferido para las columnas de la tabla.

    Returns:
        Ruta del archivo PNG guardado.

    Raises:
        ValueError: Si no hay resultados para exportar.
    """
    if not results:
        raise ValueError("No hay resultados para exportar como tabla.")

    # Import local: matplotlib solo se necesita si realmente se exporta la imagen.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_to_render = list(results[:max_rows]) if max_rows is not None else list(results)

    # Se usa la unión de claves para soportar filas con distinta estructura.
    if exclude_columns is None:
        exclude_columns = DEFAULT_COMPARE_TABLE_EXCLUDE_COLUMNS

    excluded = set(exclude_columns or [])
    available_headers = {
        key
        for row in rows_to_render
        for key, value in row.items()
        if key not in excluded and _is_simple_table_value(value)
    }

    headers: List[str] = []
    used = set()

    if header_order is None:
        header_order = DEFAULT_COMPARE_TABLE_HEADER_ORDER

    for key in header_order:
        if key in available_headers:
            headers.append(key)
            used.add(key)

    # Las columnas no priorizadas se agregan al final en orden alfabético
    # para mantener una salida determinista.
    headers.extend(sorted(available_headers - used))

    if not headers:
        raise ValueError("No hay columnas simples para exportar como tabla.")
    
    cell_text = []
    for row in rows_to_render:
        cell_text.append([_format_table_value(row.get(header, "")) for header in headers])

    n_rows = len(cell_text)
    n_cols = max(1, len(headers))

    # El tamaño crece con filas y columnas. Esto evita imágenes minúsculas cuando
    # la tabla es ancha o larga.
    fig_width = max(10, n_cols * 2.2)
    fig_height = max(3, n_rows * 0.45 + 1.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title)

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return out_path


def save_compare_run(
    *,
    run_tag: str,
    config_snapshot: Dict[str, Any],
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
    file_name: Optional[str] = None,
    export_results_table: bool = True,
    table_file_name: Optional[str] = None,
    max_rows_table: Optional[int] = 50,
) -> Path:
    """Guarda una corrida de comparación/evaluación en formato JSON.

    Args:
        run_tag: Etiqueta lógica de la corrida, por ejemplo ``"run01"``.
        config_snapshot: Configuración usada para ejecutar la corrida.
        summary: Resumen agregado de métricas.
        results: Métricas individuales por instancia, modelo o experimento.
        file_name: Nombre opcional del archivo JSON. Si es ``None``, se usa
            ``run_tag``.
        export_results_table: Si es True, también exporta una tabla PNG con
            los resultados.
        table_file_name: Nombre opcional del archivo PNG. Si es ``None``, se
            usa ``"compare{run_tag}_results.png"``.
        max_rows_table: Máximo de filas a renderizar en la tabla PNG.

    Returns:
        Ruta absoluta del archivo JSON guardado.

    Raises:
        ValueError: Si ``run_tag`` o ``file_name`` están vacíos.
    """
    clean_run_tag = str(run_tag).strip()
    if not clean_run_tag:
        raise ValueError("run_tag no puede estar vacío.")

    output_name = ensure_json_suffix(file_name or clean_run_tag)
    path = build_compare_results_path(output_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_tag": clean_run_tag,
        "config_snapshot": dict(config_snapshot),
        "summary": dict(summary),
        "results": list(results),
    }

    # Escritura atómica: primero escribe un temporal y luego reemplaza el destino.
    # Esto reduce el riesgo de dejar un JSON corrupto si el proceso se interrumpe.
    dump_json(path, payload)

    if export_results_table and results:
        png_name = ensure_png_suffix(table_file_name or f"compare{clean_run_tag}_results")
        png_path = build_compare_results_path(png_name)

        _save_results_table_image(
            results=results,
            out_path=png_path,
            title=f"Resultados de {clean_run_tag}",
            max_rows=max_rows_table,
        )

    return path