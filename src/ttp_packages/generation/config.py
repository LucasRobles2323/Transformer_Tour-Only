#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/generation/config.py

"""Configuración general para la generación de instancias TTP.

Este módulo contiene parámetros de alto nivel usados por
``generation.instance_generator``: número de ciudades, cantidad de ítems,
tipo de correlación profit-peso, velocidades, rango de coordenadas y opciones
para estimar el factor de renta ``R``.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class CorrelationType(Enum):
    """Tipos de correlación profit-peso usados al generar ítems.

    Attributes:
        UNCORRELATED: Profit y peso se generan de forma independiente.
        UNCORRELATED_SIMILAR_WEIGHTS: Pesos casi iguales con profits aleatorios.
        BOUNDED_STRONGLY_CORR: Profit correlacionado linealmente con el peso.
    """

    UNCORRELATED = "uncorrelated"
    UNCORRELATED_SIMILAR_WEIGHTS = "uncorrelated_similar_weights"
    BOUNDED_STRONGLY_CORR = "bounded_strongly_corr"


# ---------------------------------------------------------------------------
# Identificadores internos
# ---------------------------------------------------------------------------

AUX_INSTANCE_NAME = "_aux_instance"


# ---------------------------------------------------------------------------
# Parámetros numéricos de generación
# ---------------------------------------------------------------------------

DEFAULT_ITEM_MIN_VAL = 1
DEFAULT_ITEM_MAX_VAL = 1000
SIMILAR_WEIGHTS_MIN = 100000
SIMILAR_WEIGHTS_MAX = 100100
STRONGLY_CORR_BONUS = 200
CAPACITY_DIVISOR = 6.0
DEFAULT_COORD_RANGE = (0, 100000, 0, 100000)
COORD_ROUNDING_PRECISION = 4
MAX_ATTEMPTS_MULTIPLIER = 100


# ---------------------------------------------------------------------------
# Modos de estimación del factor de renta R
# ---------------------------------------------------------------------------

RENT_MODE_EXACT = "exact"
RENT_MODE_FAST = "fast"

RENT_TOUR_MODE_NN_2OPT = "nn_2opt"
RENT_TOUR_MODE_ORTOOLS = "ortools"

KNAP_MODE_AUTO = "auto"
KNAP_MODE_DP = "dp"
KNAP_MODE_FPTAS = "fptas"
KNAP_MODE_SCIP = "scip"


@dataclass
class InstanceGeneratorParams:
    """Parámetros para generar una instancia sintética TTP.

    Attributes:
        n_cities: Número total de ciudades. Debe ser al menos 3.
        item_factor: Cantidad de ítems generados por cada ciudad distinta del
            depósito.
        weight_category: Factor usado para calcular la capacidad de la mochila.
        corr_type_value: Tipo de correlación entre profit y peso.
        compute_R_mode: Modo de estimación de renta. Valores esperados:
            ``"exact"`` o ``"fast"``.
        compute_R_eps: Tolerancia usada por aproximaciones de mochila.
        compute_R_tour_mode: Método para construir el tour proxy usado al
            estimar ``R``.
        compute_R_tsp_time_limit_s: Límite de tiempo para el TSP proxy.
        compute_R_2opt_iters: Iteraciones máximas de mejora 2-opt.
        compute_R_knapsack_mode: Método de mochila usado en modo rápido.
        compute_R_knapsack_time_limit_s: Límite de tiempo para mochila.
        compute_R_knapsack_gap: Gap relativo permitido en SCIP.
        compute_R_knapsack_hide_output: Si es True, oculta salida de SCIP.
        coord_range: Rango espacial ``(min_x, max_x, min_y, max_y)``.
        min_speed: Velocidad mínima del ladrón con mochila cargada.
        max_speed: Velocidad máxima del ladrón sin carga.
        verbose: Si es True, registra detalles durante la generación.
    """

    n_cities: int = 50
    item_factor: int = 5
    weight_category: int = 6
    corr_type_value: str = CorrelationType.BOUNDED_STRONGLY_CORR.value

    # R siempre se calcula al generar una instancia.
    compute_R_mode: str = RENT_MODE_FAST
    compute_R_eps: float = 0.05

    # Configuración del tour proxy para estimar R.
    compute_R_tour_mode: str = RENT_TOUR_MODE_ORTOOLS
    compute_R_tsp_time_limit_s: float = 2.0
    compute_R_2opt_iters: int = 200

    # Configuración de la mochila proxy para estimar R.
    compute_R_knapsack_mode: str = KNAP_MODE_SCIP
    compute_R_knapsack_time_limit_s: float = 2.0
    compute_R_knapsack_gap: float = 0.05
    compute_R_knapsack_hide_output: bool = True

    coord_range: Tuple[int, int, int, int] = DEFAULT_COORD_RANGE
    min_speed: float = 0.1
    max_speed: float = 1.0
    verbose: bool = False