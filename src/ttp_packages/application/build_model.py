#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/build_model.py

"""Construcción de modelos TTP desde la capa de aplicación.

Este módulo instancia modelos neuronales usando parámetros de arquitectura y los
mueve al device solicitado o al device por defecto del proyecto.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from . import config as cfg

from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.infrastructure.runtime import get_default_device
from src.ttp_packages.modeling.ttp_model import TTPModel


logger = setup_logger(__name__)


def instantiate_ttp_model(
    params: cfg.TTPArchitectureParams = cfg.DEFAULT_MODEL_PARAMS,
    device: Optional[Union[torch.device, str]] = None,
    dtype: torch.dtype = torch.float32,
    eval_mode: bool = True,
) -> TTPModel:
    """Instancia un ``TTPModel`` y lo mueve al device objetivo.

    Args:
        params: Parámetros de arquitectura del modelo.
        device: Device destino. Si es ``None``, usa el device por defecto del
            proyecto.
        dtype: Tipo de dato de los tensores del modelo.
        eval_mode: Si es ``True``, deja el modelo en modo evaluación.

    Returns:
        Modelo instanciado, movido al device y configurado según ``eval_mode``.
    """
    model = TTPModel(params=params)

    if device is None:
        device = get_default_device()

    # Normaliza strings y objetos torch.device antes de mover el modelo.
    model = model.to(device=torch.device(device), dtype=dtype)

    if eval_mode:
        model.eval()

    return model