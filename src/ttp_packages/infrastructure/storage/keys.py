#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/infrastructure/storage/keys.py

"""Constantes compartidas para almacenamiento de modelos, runs y métricas.

Este módulo centraliza claves usadas en JSON, checkpoints de PyTorch,
historiales de entrenamiento y artefactos de evaluación. Usar constantes evita
errores por strings repetidos en distintos módulos.
"""

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

RUN_ID_WIDTH = 2
DEFAULT_DPI = 160


# ---------------------------------------------------------------------------
# Claves de artefactos de modelo / entrenamiento
# ---------------------------------------------------------------------------

KEY_MODEL_ID = "model_id"
KEY_RUN_TAG = "run_tag"
KEY_CHECKPOINT_FILE = "checkpoint_file"
KEY_HISTORY_FILE = "history_file"
KEY_MODEL_PARAMS = "model_params"
KEY_TRAIN_PARAMS = "train_params"
KEY_STATE_DICT = "state_dict"
KEY_TORCH_VERSION = "torch_version"
KEY_HISTORY = "history"
KEY_SUMMARY = "summary"
KEY_RUNS = "runs"


# ---------------------------------------------------------------------------
# Claves de métricas en historial
# ---------------------------------------------------------------------------

KEY_EPOCH = "epoch"
KEY_TRAIN_LOSS = "train_loss"
KEY_VAL_LOSS = "val_loss"
KEY_TRAIN_ACC = "train_acc"
KEY_VAL_ACC = "val_acc"