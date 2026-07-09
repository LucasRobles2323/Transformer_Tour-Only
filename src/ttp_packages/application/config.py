#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/config.py

"""Configuración compartida para workflows de aplicación.

Este módulo centraliza defaults de generación, modelo, entrenamiento, solver,
evaluación y ejecución de workflows.

También expone tipos/configuraciones usados por los scripts CLI para mantener la
regla ``scripts -> application``.
"""

from __future__ import annotations

import random
from typing import Optional

from src.ttp_packages.generation.config import (
    DEFAULT_COORD_RANGE,
    CorrelationType,
    InstanceGeneratorParams,  # Re-export para scripts que construyen inst_params.
)
from src.ttp_packages.hpo.config import (  # Re-export para scripts/optuna_tour.py.
    OptunaModelSearchSpace,
    OptunaSearchConfig,
    OptunaTrainSearchSpace,
    OptunaWorkflowParams,
)
from src.ttp_packages.infrastructure.logging import setup_logger  # Re-export para scripts/utils.py.
from src.ttp_packages.modeling.config import TTPArchitectureParams  # Re-export para scripts/fit_tour_model.py.
from src.ttp_packages.optimization.classical.ttp.cs2sa_r.config import (
    NO_IMPROVE_PATIENCE,
    RESTART_MODE_FULL,
)
from src.ttp_packages.training.config import TrainingParams  # Re-export para scripts/fit_tour_model.py.


def resolve_seed(seed: Optional[int]) -> int:
    """Resuelve una semilla explícita o genera una aleatoria.

    Args:
        seed: Semilla opcional entregada por el usuario.

    Returns:
        Semilla entera para reproducibilidad del workflow.
    """
    if seed is not None:
        return int(seed)

    return random.SystemRandom().randrange(0, 2**63)


# ============================================================================
# Defaults generales de aplicación
# ============================================================================

DEFAULT_INST_PARAMS = InstanceGeneratorParams(
    n_cities=50,
    item_factor=5,
    weight_category=6,
    corr_type_value=CorrelationType.BOUNDED_STRONGLY_CORR,
    coord_range=DEFAULT_COORD_RANGE,
    min_speed=0.1,
    max_speed=1.0,
    verbose=False,
)

DEFAULT_MODEL_PARAMS = TTPArchitectureParams(
    d_model=64,
    n_heads=4,
    n_layers=3,
    d_ff=128,
    dropout=0.1,
    sinkhorn_iter=80,
    sink_tau=0.1,
    coupling_iters=1,
    node_feature_set = "basic",
    edge_feature_mode = "none",
)

DEFAULT_TRAIN_PARAMS = TrainingParams(
    seed=123,
    val_frac=0.20,
    batch_size=16,
    shuffle_train=True,
    num_workers=0,
    epochs=200,
    lr=5e-4,
    weight_decay=3e-4,
    grad_clip_norm=1.0,
    use_amp=None,
    mask_mode="dense",
    knn_k=10,
    allow_self=False,
    sym=True,
    compute_dist_matrix=True,
    apply_mask_to_loss=True,
    sinkhorn_nll_weight=0.0,
    device=None,
    verbose=True,
    patience=10,
    min_delta=1e-4,
    overfit_patience=5,
    overfit_min_delta=1e-4,
)


# ============================================================================
# Verbose, solver, evaluación, plots y ejecución
# ============================================================================

VERBOSE_DATA_WORK_MAIN: bool = True
VERBOSE_EVALUATE_WORK_MAIN: bool = True
VERBOSE_TRAIN_WORK_MAIN: bool = True

VERBOSE_SOLVER_CS2SAR: bool = False
VERBOSE_SAMPLE_FORMATTING: bool = False
VERBOSE_STORAGE_IO: bool = True

DEFAULT_MODE_RESTART: str = RESTART_MODE_FULL
DEFAULT_NO_IMPROVE_PATIENCE: int = NO_IMPROVE_PATIENCE

DEFAULT_SOLVER_VERBOSE_SECTIONS = None
DEFAULT_SOLVER_VERIFY_INTEGRITY: bool = False

DEFAULT_EVAL_TIME_BUDGET_S: float = 30.0
DEFAULT_EVAL_N_RESTARTS: int = 3

DEFAULT_PLOT_DPI: int = 160

DEFAULT_SEED: Optional[int] = None
DEFAULT_VERBOSE: bool = True