#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/application/load_instance.py

"""Carga y generación de instancias TTP desde la capa de aplicación.

Este módulo ofrece funciones de alto nivel para obtener instancias reales desde
archivo o instancias sintéticas generadas por configuración.
"""

from __future__ import annotations

from . import config as cfg

from src.ttp_packages.domain.instance import TTPInstance
from src.ttp_packages.generation.instance_generator import generate_ttp_instance
from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.infrastructure.storage.instance_io import load_instance


logger = setup_logger(__name__)


def get_real_inst(file_name: str) -> TTPInstance:
    """Carga una instancia TTP real desde archivo.

    Args:
        file_name: Ruta o nombre del archivo de instancia.

    Returns:
        Instancia TTP cargada en memoria.
    """
    return load_instance(fname=file_name)


def get_generated_inst(
    inst_params: cfg.InstanceGeneratorParams = cfg.DEFAULT_INST_PARAMS,
) -> TTPInstance:
    """Genera una instancia TTP sintética.

    Args:
        inst_params: Parámetros de generación de la instancia.

    Returns:
        Instancia TTP generada y lista para usarse.
    """
    return generate_ttp_instance(params=inst_params)