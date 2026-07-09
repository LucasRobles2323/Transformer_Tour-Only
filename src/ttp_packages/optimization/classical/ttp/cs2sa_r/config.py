#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/ttp/cs2sa_r/config.py

"""Constantes del solver CS2SA-R.

Este módulo centraliza parámetros de reinicio, logging segmentado,
inicialización, KRP y TSKP.
"""

# ---------------------------------------------------------------------------
# Configuración general CS2SA-R
# ---------------------------------------------------------------------------

RESTART_MODE_FULL = "full"
RESTART_MODE_LIGHT = "light"

NO_IMPROVE_PATIENCE = 3
START_CITY = 0

# Umbral usado para activar estrategias especiales en instancias muy grandes.
VERY_LARGE_INSTANCE_THRESHOLD = 18512

# Solo se usa cuando restart_mode == "light".
RESTART_NOISE = 0.03


# ---------------------------------------------------------------------------
# Verbose / logging segmentado
# ---------------------------------------------------------------------------

DEFAULT_VERBOSE_SECTIONS = {}

VERBOSE_SECTION_CYCLES = "cycles"
VERBOSE_SECTION_INITIAL = "initial"
VERBOSE_SECTION_TSKP = "tskp"
VERBOSE_SECTION_KRP = "krp"
VERBOSE_SECTION_INTEGRITY = "integrity"

VERBOSE_ALIAS_STEPS = "steps"
VERBOSE_ALIAS_ALL = "all"
VERBOSE_ALIAS_NONE = "none"

VERBOSE_SECTIONS_ALL = (
    VERBOSE_SECTION_CYCLES,
    VERBOSE_SECTION_INITIAL,
    VERBOSE_SECTION_TSKP,
    VERBOSE_SECTION_KRP,
    VERBOSE_SECTION_INTEGRITY,
)

VERBOSE_SECTIONS_STEPS = (
    VERBOSE_SECTION_INITIAL,
    VERBOSE_SECTION_TSKP,
    VERBOSE_SECTION_KRP,
)


# ---------------------------------------------------------------------------
# Initializer / LK heuristic
# ---------------------------------------------------------------------------

INIT_ELIMINATION_CITY_CUTOFF = 30000
INIT_BOOST_CITY_CUTOFF = 200

# Métodos esperados por lk_heuristic: "lk2_improve" o "lk1_improve".
LK_DEFAULT_SOLUTION_METHOD = "lk2_improve"
LK_DEFAULT_LOGGING_LEVEL = 50

# Nombres usados para archivos temporales TSPLIB.
TSPLIB_TEMP_INPUT_FILENAME = "inst.tsp"
TSPLIB_TEMP_OUTPUT_FILENAME = "inst_solution.tsp"

# Tokens usados al escribir o parsear archivos TSPLIB.
TSPLIB_NAME_PREFIX = "NAME:"
TSPLIB_TYPE_TSP = "TYPE: TSP"
TSPLIB_EDGE_WEIGHT_TYPE_EUC_2D = "EDGE_WEIGHT_TYPE: EUC_2D"
TSPLIB_NODE_COORD_SECTION = "NODE_COORD_SECTION"
TSPLIB_EOF = "EOF"


# ---------------------------------------------------------------------------
# KRP Optimizer
# ---------------------------------------------------------------------------

KRP_TABS = 1.0
KRP_T0 = 100.0
KRP_ALPHA = 0.95

# Puntos de interpolación usados para estimar el número de trials de KRP.
KRP_X_SPLINE = (50, 204, 609, 1147, 8034, 38875, 105318, 253568, 338090)
KRP_Y_SPLINE = (1000.0, 500.0, 100.0, 50.0, 10.0, 1.0, 0.04, 0.03, 0.03)
KRP_MAX_SPLINE_VAL = 57872.0


# ---------------------------------------------------------------------------
# TSKP Optimizer
# ---------------------------------------------------------------------------

TSKP_DEFAULT_IMPROVEMENT_THRESHOLD = -0.1

TSKP_MEDIUM_INSTANCE_MIN_ITEMS = 100000
TSKP_MEDIUM_INSTANCE_IMPROVEMENT_THRESHOLD = -10.0

TSKP_LARGE_INSTANCE_MIN_CITIES = 50000
TSKP_LARGE_INSTANCE_IMPROVEMENT_THRESHOLD = -1000.0


# ---------------------------------------------------------------------------
# Verificación / UI
# ---------------------------------------------------------------------------

SEPARATOR = "-" * 50