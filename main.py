#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


DEFAULT_DATASET_CONFIG = Path("configs/datasets/generate_dataset.json")
DEFAULT_MERGE_DATASET_CONFIG = Path("configs/datasets/merge_dataset.json")
DEFAULT_BENCHMARK_CONFIG = Path("configs/evaluation/benchmark_cs2sar_vs_tsp.json")
DEFAULT_EVALUATE_SOLVER_CONFIG = Path("configs/evaluation/evaluate_solver_cs2sar.json")
DEFAULT_TRAIN_CONFIG = Path("configs/training/fit_tour_model.json")
DEFAULT_OPTUNA_CONFIG = Path("configs/optuna/optuna_tour.json")
DEFAULT_EVALUATE_MODEL_CONFIG = Path("configs/evaluation/evaluate_model_with_solvers.json")


def _resolve_command(command: str) -> tuple[Callable[[Path], None], Path]:
    if command == "generate-data":
        from scripts.gen_dataset import run as runner
        return runner, DEFAULT_DATASET_CONFIG

    if command == "merge-data":
        from scripts.merge_data import run as runner
        return runner, DEFAULT_MERGE_DATASET_CONFIG

    if command == "benchmark":
        from scripts.benchmark_cs2sar_vs_tsp import run as runner
        return runner, DEFAULT_BENCHMARK_CONFIG

    if command == "evaluate-solver":
        from scripts.evaluate_solver_cs2sar import run as runner
        return runner, DEFAULT_EVALUATE_SOLVER_CONFIG

    if command == "fit-model":
        from scripts.fit_tour_model import run as runner
        return runner, DEFAULT_TRAIN_CONFIG

    if command == "optuna":
        from scripts.optuna_tour import run as runner
        return runner, DEFAULT_OPTUNA_CONFIG
    
    if command == "evaluate-model":
        from scripts.evaluate_model_with_solvers import run as runner
        return runner, DEFAULT_EVALUATE_MODEL_CONFIG

    raise ValueError(
        f"Acción desconocida: {command}. "
        f"Opciones válidas: "
        f"{['generate-data', 'merge-data', 'benchmark', 'evaluate-solver', 'evaluate-model', 'fit-model', 'optuna']}"
    )


def main(command: str = "optuna") -> None:
    # Obtiene la función a ejecutar y su JSON predeterminado automáticamente
    runner, config_path = _resolve_command(command)
    
    print(f"-> Usando configuración: {config_path}")
    runner(config_path)


if __name__ == "__main__":
    import multiprocessing as mp

    # Inocuo si no estás congelando el ejecutable; útil para multiprocessing en Windows
    mp.freeze_support() 

    # Configuración de los argumentos de la consola (Mucho más limpio)
    parser = argparse.ArgumentParser(description="Script principal para TTP Transformer Project")
    
    parser.add_argument(
        "command", 
        type=str, 
        nargs="?", 
        default="merge-data", 
        choices=[
            'generate-data', 'merge-data', 'benchmark', 
            'evaluate-solver', 'evaluate-model', 'fit-model', 'optuna'
        ],
        help="El comando a ejecutar. El script buscará su JSON automáticamente."
    )

    args = parser.parse_args()

    try:
        print(f"[MAIN] Comienza. Ejecutando: {args.command}\n")
        main(command=args.command)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrumpido por usuario.")
    finally:
        print(f"\n[MAIN] Termina ")