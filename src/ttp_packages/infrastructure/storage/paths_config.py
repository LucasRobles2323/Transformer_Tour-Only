#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/paths_config.py

"""Configuración centralizada de rutas base del proyecto.

Este módulo define las carpetas principales usadas por la capa de almacenamiento:
instancias, datasets, modelos entrenados, parámetros, historiales, gráficos,
resultados de comparación y estudios Optuna.
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT_DIR / "data"

INSTANCES_DIR = DATA_DIR / "instances"
TRAIN_DATA_DIR = DATA_DIR / "train_data"
OPTUNA_DATA_DIR = DATA_DIR / "optuna"
RESULTS_COMPARE_DIR = DATA_DIR / "results_compare"

TRAINED_MODEL_DIR = DATA_DIR / "trained_models"
TRAINED_MODEL_PARAMS_DIR = DATA_DIR / "trained_models_params"
TRAINED_MODEL_HISTORY_DIR = DATA_DIR / "trained_history_runs"
TRAINED_MODEL_PLOTS_DIR = DATA_DIR / "training_plots"