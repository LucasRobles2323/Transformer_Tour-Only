#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/generate_dataset.py

"""Workflow de generación y extensión de datasets tensoriales TTP.

Este módulo orquesta la creación de instancias sintéticas, su resolución mediante
CS2SA-R, la conversión a muestras tensoriales y la persistencia incremental del
dataset en disco.
"""

from __future__ import annotations

import random
import re
import signal
from pathlib import Path
from time import perf_counter
from types import FrameType
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from . import config as cfg
from .solve_instance import solve_with_cs2sar

from src.ttp_packages.generation.instance_generator import generate_ttp_instance
from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.infrastructure.storage.dataset_io import (
    append_samples_to_payload,
    prepare_dataset_for_append,
    save_dataset,
)
from src.ttp_packages.ml_data.representation.instance_solution import (
    instance_solution_to_sample,
)


logger = setup_logger(__name__)


DEFAULT_EXTENSION = ".pt"
DATASET_PARAMS_REGEX = r".*_n\d+_fi\d+_wc\d+_.+"
_PARAMS_RE = re.compile(DATASET_PARAMS_REGEX)

STOP_REQUESTED = False


def _handle_sigint(signum: int, frame: FrameType | None) -> None:
    """Maneja Ctrl+C con detención segura después de la muestra actual.

    Args:
        signum: Señal recibida por el proceso.
        frame: Frame activo al momento de recibir la señal.

    Raises:
        KeyboardInterrupt: Si el usuario presiona Ctrl+C por segunda vez.
    """
    global STOP_REQUESTED

    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n[STOP] Ctrl+C recibido. Se detendrá después de la muestra actual.")
    else:
        print("\n[STOP] Segundo Ctrl+C. Abortando inmediatamente.")
        raise KeyboardInterrupt


def expand_dataset_filename(
    base_name: str,
    n_cities: int,
    item_factor: int,
    weight_category: int,
    corr_type: cfg.CorrelationType | str,
) -> str:
    """Expande un nombre de dataset incorporando parámetros TTP.

    Args:
        base_name: Nombre base del archivo.
        n_cities: Número de ciudades.
        item_factor: Cantidad de ítems por ciudad distinta del depósito.
        weight_category: Categoría usada para calcular capacidad.
        corr_type: Tipo de correlación profit-peso.

    Returns:
        Nombre de archivo con extensión ``.pt`` y parámetros embebidos.
    """
    path = Path(base_name)
    suffix = path.suffix if path.suffix else DEFAULT_EXTENSION
    stem = path.stem

    corr_str = getattr(corr_type, "value", str(corr_type))
    corr_str = str(corr_str).strip().replace(" ", "_")

    # Evita duplicar parámetros cuando el nombre ya fue expandido previamente.
    if _PARAMS_RE.match(stem):
        new_stem = stem
    else:
        new_stem = (
            f"{stem}"
            f"_n{n_cities}"
            f"_fi{item_factor}"
            f"_wc{weight_category}"
            f"_{corr_str}"
        )

    return str(path.with_name(new_stem + suffix))


def generate_samples(
    n_samples: int = 100,
    print_offset: int = 0,
    inst_params: cfg.InstanceGeneratorParams = cfg.DEFAULT_INST_PARAMS,
    solver_time_budget_s: float = 60.0,
    solver_restart_mode: str = cfg.DEFAULT_MODE_RESTART,
    solver_no_improve_patience: int = cfg.DEFAULT_NO_IMPROVE_PATIENCE,
    verbose: bool = cfg.VERBOSE_DATA_WORK_MAIN,
    verbose_debug: bool = False,
    solver_verbose_sections: Optional[Iterable[str] | str] = (
        cfg.DEFAULT_SOLVER_VERBOSE_SECTIONS
    ),
    solver_verify_integrity: bool = cfg.DEFAULT_SOLVER_VERIFY_INTEGRITY,
    verbose_format: bool = cfg.VERBOSE_SAMPLE_FORMATTING,
    log_fn: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = cfg.DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """Genera muestras individuales ``instancia + solución``.

    Cada muestra se produce generando una instancia TTP sintética, resolviéndola
    con CS2SA-R y convirtiendo el par ``(instancia, solución)`` al formato
    tensorial esperado por el pipeline de entrenamiento.

    Args:
        n_samples: Cantidad de muestras a generar.
        print_offset: Offset usado para numerar el progreso cuando se expande un
            dataset existente.
        inst_params: Parámetros de generación de instancias.
        solver_time_budget_s: Presupuesto de tiempo por instancia para CS2SA-R.
        solver_restart_mode: Modo de reinicio del solver.
        solver_no_improve_patience: Paciencia sin mejora.
        verbose: Si es ``True``, registra progreso resumido.
        verbose_debug: Si es ``True``, registra tiempos detallados por etapa.
        solver_verbose_sections: Secciones verbosas del solver.
        solver_verify_integrity: Si es ``True``, valida integridad de soluciones.
        verbose_format: Si es ``True``, registra detalles del formateo a tensores.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.
        seed: Semilla base. Si es ``None``, se genera una aleatoria.

    Returns:
        Lista de muestras listas para convertir o anexar a un payload.

    Raises:
        ValueError: Si ``solver_time_budget_s`` no es positivo.
    """
    if log_fn is None:
        log_fn = logger.info

    if solver_time_budget_s <= 0:
        raise ValueError("solver_time_budget_s debe ser > 0")

    n_samples = int(n_samples)
    if n_samples < 1:
        return []

    base_seed = cfg.resolve_seed(seed)

    random.seed(base_seed)
    torch.manual_seed(base_seed)

    base = max(0, int(print_offset))
    total_after = base + n_samples

    samples: List[Dict[str, Any]] = []
    global STOP_REQUESTED

    for sample_index in range(n_samples):
        if STOP_REQUESTED:
            log_fn("[GEN_SAMPLE] Stop solicitado. Cortando generación.")
            break

        global_index = base + sample_index + 1
        instance_seed = base_seed + 100003 * sample_index + 11
        solver_seed = base_seed + 100003 * sample_index + 29

        if verbose_debug:
            log_fn(
                f"\n[GEN_SAMPLE][{global_index}/{total_after}] START | "
                f"solver_time_budget_s={float(solver_time_budget_s):.2f} | "
                f"restart_mode={solver_restart_mode} | "
                f"no_improve_patience={int(solver_no_improve_patience)} | "
                f"inst_seed={instance_seed} | solver_seed={solver_seed}"
            )

        sample_start = perf_counter()

        # Genera la instancia con un RNG local para que cada muestra sea reproducible.
        instance_start = perf_counter()
        rng = random.Random(instance_seed)
        instance = generate_ttp_instance(
            params=inst_params,
            rng=rng,
            log_fn=log_fn,
        )
        generation_time = perf_counter() - instance_start

        if verbose_debug:
            log_fn(
                f"[GEN_SAMPLE][{global_index}/{total_after}] INSTANCE | "
                f"gen_time={generation_time:.3f}s | "
                f"N={int(instance.n_cities)} | M={int(instance.m_items)} | "
                f"capacity={float(instance.capacity):.3f} | "
                f"R={float(instance.rent_per_time):.6f}"
            )

        # Resuelve la instancia con CS2SA-R usando una semilla separada.
        solver_start = perf_counter()
        solution = solve_with_cs2sar(
            instance,
            time_to_solve=float(solver_time_budget_s),
            restart_mode=solver_restart_mode,
            no_improve_patience=solver_no_improve_patience,
            seed=solver_seed,
            verbose_sections=solver_verbose_sections,
            verify_integrity=solver_verify_integrity,
            log_fn=log_fn,
        )
        solver_time = perf_counter() - solver_start

        if verbose_debug:
            log_fn(
                f"[GEN_SAMPLE][{global_index}/{total_after}] SOLVER | "
                f"requested_budget={float(solver_time_budget_s):.2f}s | "
                f"wall_time={solver_time:.3f}s | "
                f"sol_time={float(getattr(solution, 'time', float('nan'))):.3f} | "
                f"sol_profit={float(getattr(solution, 'profit', float('nan'))):.3f} | "
                f"sol_obj={float(getattr(solution, 'objective', float('nan'))):.3f} | "
                f"tour_len={len(getattr(solution, 'tour', [])) if getattr(solution, 'tour', None) is not None else -1} | "
                f"packing_len={len(getattr(solution, 'packing', [])) if getattr(solution, 'packing', None) is not None else -1}"
            )

        # Recalcula métricas derivadas antes de convertir la solución a tensores.
        recompute_start = perf_counter()
        recomputed = False
        if (
            getattr(solution, "tour", None) is not None
            and getattr(solution, "packing", None) is not None
            and hasattr(solution, "compute_benefit")
        ):
            solution.compute_benefit()
            recomputed = True
        recompute_time = perf_counter() - recompute_start

        if verbose_debug:
            log_fn(
                f"[GEN_SAMPLE][{global_index}/{total_after}] RECOMPUTE | "
                f"used={recomputed} | time={recompute_time:.3f}s"
            )

        # Convierte instancia y solución al formato tensorial consumido por ML.
        format_start = perf_counter()
        sample = instance_solution_to_sample(
            inst=instance,
            sol=solution,
            verbose=verbose_format,
            log_fn=log_fn,
        )
        format_time = perf_counter() - format_start

        samples.append(sample)
        total_elapsed = perf_counter() - sample_start

        if STOP_REQUESTED:
            log_fn("[GEN_SAMPLE] Stop solicitado tras completar la muestra actual.")
            break

        meta = sample["meta"]
        teacher = sample["teacher"]

        if verbose:
            log_fn(
                f"[{global_index}/{total_after}] "
                f"objective={teacher['objective']:.3f} | "
                f"time={total_elapsed:.3f}s | "
                f"profit={teacher['profit']:.3f} "
                f"time={teacher['time']:.3f} | "
                f"N={meta['n_cities']} M={meta['m_items']}"
            )

        if verbose_debug:
            log_fn(
                f"[GEN_SAMPLE][{global_index}/{total_after}] END | "
                f"total={total_elapsed:.3f}s | "
                f"gen={generation_time:.3f}s | solver={solver_time:.3f}s | "
                f"recompute={recompute_time:.3f}s | format={format_time:.3f}s | "
                f"obj={teacher['objective']:.3f} | "
                f"profit={teacher['profit']:.3f} | "
                f"time={teacher['time']:.3f} | "
                f"N={meta['n_cities']} M={meta['m_items']}"
            )

    return samples


def generate_tensor_dataset(
    file_name: str = "data00.pt",
    n_new_samples: int = 100,
    inst_params: cfg.InstanceGeneratorParams = cfg.DEFAULT_INST_PARAMS,
    solver_time_budget_s: float = 60.0,
    solver_restart_mode: str = cfg.DEFAULT_MODE_RESTART,
    solver_no_improve_patience: int = cfg.DEFAULT_NO_IMPROVE_PATIENCE,
    verbose: bool = cfg.VERBOSE_DATA_WORK_MAIN,
    verbose_storage: bool = cfg.VERBOSE_STORAGE_IO,
    solver_verbose_sections: Optional[Iterable[str] | str] = (
        cfg.DEFAULT_SOLVER_VERBOSE_SECTIONS
    ),
    solver_verify_integrity: bool = cfg.DEFAULT_SOLVER_VERIFY_INTEGRITY,
    verbose_format: bool = cfg.VERBOSE_SAMPLE_FORMATTING,
    log_fn: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = cfg.DEFAULT_SEED,
) -> Dict[str, Any]:
    """Genera o expande un dataset tensorial TTP.

    Args:
        file_name: Nombre base del archivo de dataset.
        n_new_samples: Cantidad de muestras nuevas a generar.
        inst_params: Parámetros de generación de instancias TTP.
        solver_time_budget_s: Presupuesto de tiempo por instancia para CS2SA-R.
        solver_restart_mode: Modo de reinicio del solver.
        solver_no_improve_patience: Paciencia sin mejora antes de reiniciar.
        verbose: Si es ``True``, registra progreso general del workflow.
        verbose_storage: Si es ``True``, registra operaciones de almacenamiento.
        solver_verbose_sections: Secciones verbosas activas para el solver.
        solver_verify_integrity: Si es ``True``, verifica integridad de soluciones.
        verbose_format: Si es ``True``, registra detalles de formateo de muestras.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.
        seed: Semilla base para reproducibilidad.

    Returns:
        Payload actualizado del dataset.

    Raises:
        ValueError: Si ``solver_time_budget_s`` o ``n_new_samples`` son inválidos.
        KeyboardInterrupt: Si se solicita detención segura con Ctrl+C.
    """
    if log_fn is None:
        log_fn = logger.info

    global STOP_REQUESTED
    STOP_REQUESTED = False

    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        run_seed = cfg.resolve_seed(seed)

        if solver_time_budget_s <= 0:
            raise ValueError("solver_time_budget_s debe ser > 0")

        n_new_samples = int(n_new_samples)
        if n_new_samples < 1:
            raise ValueError("n_new_samples debe ser >= 1")

        # El nombre expandido codifica dimensiones y tipo de correlación del dataset.
        expanded_name = expand_dataset_filename(
            base_name=file_name,
            n_cities=inst_params.n_cities,
            item_factor=inst_params.item_factor,
            weight_category=inst_params.weight_category,
            corr_type=inst_params.corr_type_value,
        )

        m_items = int(inst_params.item_factor * (inst_params.n_cities - 1))

        payload, offset = prepare_dataset_for_append(
            file_name=expanded_name,
            n_cities=inst_params.n_cities,
            m_items=m_items,
            verbose=verbose_storage,
            save_empty=False,
            log_fn=log_fn,
        )

        new_samples = generate_samples(
            n_samples=n_new_samples,
            print_offset=offset,
            inst_params=inst_params,
            solver_time_budget_s=solver_time_budget_s,
            solver_restart_mode=solver_restart_mode,
            solver_no_improve_patience=solver_no_improve_patience,
            verbose=verbose,
            solver_verbose_sections=solver_verbose_sections,
            solver_verify_integrity=solver_verify_integrity,
            verbose_format=verbose_format,
            log_fn=log_fn,
            seed=run_seed,
        )

        payload = append_samples_to_payload(
            payload,
            new_samples,
            verbose=verbose_storage,
            log_fn=log_fn,
        )

        saved_path = save_dataset(
            expanded_name,
            payload,
            verbose=verbose_storage,
            log_fn=log_fn,
        )

        if verbose:
            log_fn(f"[DATASET COMPLETE] Archivo: {saved_path}")

        if STOP_REQUESTED:
            log_fn("[STOP] Guardado completo. Saliendo del ciclo principal.")
            raise KeyboardInterrupt

        return payload

    finally:
        signal.signal(signal.SIGINT, old_handler)


def generate_tensor_dataset_work(
    *,
    file_name: str,
    n_new_samples: int,
    inst_params: cfg.InstanceGeneratorParams = cfg.DEFAULT_INST_PARAMS,
    solver_time_budget_s: float = 60.0,
    solver_restart_mode: str = cfg.DEFAULT_MODE_RESTART,
    solver_no_improve_patience: int = cfg.DEFAULT_NO_IMPROVE_PATIENCE,
    solver_verbose_sections: Optional[Iterable[str] | str] = (
        cfg.DEFAULT_SOLVER_VERBOSE_SECTIONS
    ),
    solver_verify_integrity: bool = cfg.DEFAULT_SOLVER_VERIFY_INTEGRITY,
    verbose: bool = cfg.VERBOSE_DATA_WORK_MAIN,
    verbose_storage: bool = cfg.VERBOSE_STORAGE_IO,
    verbose_format: bool = cfg.VERBOSE_SAMPLE_FORMATTING,
    log_fn: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = cfg.DEFAULT_SEED,
) -> Dict[str, Any]:
    """Genera o extiende un dataset tensorial TTP desde el workflow de aplicación.

    Si el archivo no existe, crea un payload vacío. Si ya existe, lo carga,
    valida sus dimensiones y agrega nuevas muestras.

    Args:
        file_name: Nombre base del archivo de dataset.
        n_new_samples: Cantidad de muestras nuevas.
        inst_params: Parámetros de generación de instancias.
        solver_time_budget_s: Presupuesto de tiempo por instancia.
        solver_restart_mode: Modo de reinicio del solver.
        solver_no_improve_patience: Paciencia sin mejora.
        solver_verbose_sections: Secciones verbosas del solver.
        solver_verify_integrity: Si es ``True``, valida integridad.
        verbose: Si es ``True``, registra progreso general.
        verbose_storage: Si es ``True``, registra operaciones de storage.
        verbose_format: Si es ``True``, registra formateo.
        log_fn: Función opcional de logging. Si es ``None``, usa
            ``logger.info``.
        seed: Semilla base.

    Returns:
        Payload actualizado del dataset.
    """
    if log_fn is None:
        log_fn = logger.info

    run_seed = cfg.resolve_seed(seed)

    return generate_tensor_dataset(
        file_name=file_name,
        n_new_samples=n_new_samples,
        inst_params=inst_params,
        solver_time_budget_s=solver_time_budget_s,
        solver_restart_mode=solver_restart_mode,
        solver_no_improve_patience=solver_no_improve_patience,
        solver_verbose_sections=solver_verbose_sections,
        solver_verify_integrity=solver_verify_integrity,
        verbose=verbose,
        verbose_storage=verbose_storage,
        verbose_format=verbose_format,
        log_fn=log_fn,
        seed=run_seed,
    )