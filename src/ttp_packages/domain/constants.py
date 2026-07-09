#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/domain/constants.py

"""Constantes globales del paquete domain para el problema TTP.

Este módulo centraliza valores compartidos por las entidades, la evaluación
de soluciones y las rutinas de visualización/logging.
"""

# Centinelas usados al evaluar soluciones inválidas o no factibles.
INF = float('inf')
NEG_INF = float('-inf')

# Parámetros de formato para reportes y representaciones de objetos.
DEBUG_LINE_WIDTH = 60
DECIMALS = 2