#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/ttp/cs2sa_r/krp_optimizer.py

"""Optimizador KRP para packing mediante Simulated Annealing.

Este módulo implementa la fase KRP de CS2SA-R. Dado un tour fijo y un packing
inicial, explora vecinos bit-flip usando una política de aceptación tipo
Simulated Annealing.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from typing import Optional

from src.ttp_packages.infrastructure.logging import setup_logger

from .config import (
    KRP_ALPHA,
    KRP_MAX_SPLINE_VAL,
    KRP_T0,
    KRP_TABS,
    KRP_X_SPLINE,
    KRP_Y_SPLINE,
)
from .delta_eval import incremental_bitflip, recompute_history_full


logger = setup_logger(__name__)


class KRPOptimizer:
    """Optimizador KRP basado en Simulated Annealing para packing.

    Attributes:
        math: Referencia local al módulo ``math`` usada en el cálculo de aceptación.
        rnd: Generador pseudoaleatorio inicializado con semilla.
        T_abs: Temperatura absoluta de corte.
        T0: Temperatura inicial.
        alpha: Factor de enfriamiento geométrico.
        L_function: Función que calcula el factor de trials.
        nb_trials: Cantidad base de intentos por temperatura.
    """

    def __init__(
        self,
        rnd_seed: int = 1,
        T_abs: float = KRP_TABS,
        T0: float = KRP_T0,
        alpha: float = KRP_ALPHA,
    ):
        """Inicializa el optimizador KRP.

        Args:
            rnd_seed: Semilla para el generador pseudoaleatorio.
            T_abs: Temperatura absoluta de corte.
            T0: Temperatura inicial del proceso de annealing.
            alpha: Factor de enfriamiento geométrico.
        """
        self.math = math
        self.rnd = random.Random(rnd_seed)
        self.T_abs = T_abs
        self.T0 = T0
        self.alpha = alpha
        self.L_function = self._L_piecewise_CS2SA_R_exact
        self.nb_trials = 1000

    @staticmethod
    def _L_piecewise_CS2SA_R_exact(m: int) -> float:
        """Calcula el factor de intentos mediante interpolación lineal por tramos.

        Args:
            m: Cantidad de ítems en la instancia.

        Returns:
            Multiplicador ``L`` para el número de iteraciones en Simulated
            Annealing.

        Raises:
            RuntimeError: Si no se puede determinar un factor para ``m``.
        """
        x = KRP_X_SPLINE
        y = KRP_Y_SPLINE

        if m <= x[0]:
            return KRP_MAX_SPLINE_VAL

        if m >= x[-1]:
            return y[-1]

        for i in range(len(x) - 1):
            if x[i] <= m < x[i + 1]:
                slope = (y[i] - y[i + 1]) / (x[i] - x[i + 1])
                intercept = y[i] - slope * x[i]
                return slope * m + intercept

        raise RuntimeError(f"trial factor no definido para m={m}")

    def _set_sa_trials(self, m: int) -> None:
        """Establece la cantidad base de trials para Simulated Annealing.

        Args:
            m: Cantidad de ítems en la instancia.
        """
        trial_factor = self.L_function(m)
        more_exploration_factor = 0.01

        # Ajusta la exploración según la cantidad de ítems.
        if m < 300:
            more_exploration_factor = 0.02
        elif m > 2000:
            more_exploration_factor = 0.005
        elif m > 10000:
            more_exploration_factor = 0.001

        self.nb_trials = int(round(m * trial_factor * more_exploration_factor))

    def _speed_from_weight(self, inst, weight: float) -> float:
        """Calcula la velocidad efectiva según el peso acumulado.

        Args:
            inst: Instancia TTP con velocidades y capacidad.
            weight: Peso acumulado actual.

        Returns:
            Velocidad efectiva respetando la velocidad mínima.
        """
        speed_coef = (inst.max_speed - inst.min_speed) / float(inst.capacity)
        speed = inst.max_speed - weight * speed_coef

        return max(inst.min_speed, speed)

    def _candidate_gain_after_flip(self, inst, cache, item_id: int) -> tuple[float, int]:
        """Evalúa un vecino bit-flip sin modificar la caché.

        Args:
            inst: Instancia TTP.
            cache: Caché incremental de evaluación.
            item_id: Índice del ítem a invertir en el packing.

        Returns:
            Tupla ``(candidate_gain, new_value)`` con el objetivo estimado del
            vecino y el nuevo valor binario del ítem.
        """
        if inst.distance_matrix is None:
            inst.create_distance_matrix()

        item = inst.items[item_id]
        current_value = cache.packing[item_id]
        new_value = 1 - current_value

        delta_profit = item.profit if current_value == 0 else -item.profit
        delta_weight = item.weight if current_value == 0 else -item.weight

        candidate_profit = cache.g_profit + delta_profit
        item_position = cache.pos_in_tour[item.city_id]
        candidate_time = 0.0 if item_position == 0 else cache.t_acc[item_position - 1]

        distance_matrix = inst.distance_matrix
        if distance_matrix is None:
            raise RuntimeError("No se pudo crear la matriz de distancias.")

        # El flip cambia el peso transportado desde la ciudad del ítem en adelante.
        for route_index in range(item_position, len(cache.tour)):
            candidate_weight = cache.w_acc[route_index] + delta_weight
            candidate_time += (
                float(
                    distance_matrix[
                        cache.tour[route_index],
                        cache.tour[(route_index + 1) % len(cache.tour)],
                    ]
                )
                / self._speed_from_weight(inst, candidate_weight)
            )

        candidate_gain = candidate_profit - candidate_time * inst.rent_per_time

        return candidate_gain, new_value

    def optimize(
        self,
        inst,
        tour: list[int],
        packing: list[int],
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
        deadline: Optional[float] = None,
    ) -> tuple[list[int], list[int]]:
        """Ejecuta Simulated Annealing sobre una solución de packing.

        Args:
            inst: Instancia TTP.
            tour: Tour actual asociado a la solución.
            packing: Solución binaria inicial de packing.
            verbose: Si es True, emite trazas del proceso.
            log_fn: Función de logging. Si es ``None``, usa ``logger.info``.
            deadline: Tiempo límite absoluto para interrumpir la corrida.

        Returns:
            Tupla ``(tour, packing)`` con el tour original y el mejor packing
            encontrado.
        """
        if log_fn is None:
            log_fn = logger.info

        # Caché incremental: permite aplicar flips aceptados sin recomputar todo.
        cache = recompute_history_full(inst, tour, packing)

        # Ajusta la cantidad de trials por temperatura según el número de ítems.
        self._set_sa_trials(inst.m_items)

        m_items = inst.m_items
        items = inst.items

        initial_temperature = self.T0
        cooling_factor = self.alpha
        min_temperature = self.T_abs
        temperature = initial_temperature

        current_gain = cache.G_gain
        best_gain = current_gain
        best_packing = cache.packing[:]

        rnd_randrange = self.rnd.randrange
        rnd_random = self.rnd.random
        math_exp = self.math.exp
        time_func = time.time

        temperature_iter = 0

        if verbose:
            log_fn(
                f"[KRP] init T0={initial_temperature:.4f} "
                f"alpha={cooling_factor:.4f} "
                f"T_abs={min_temperature:.4f} nb_trials={self.nb_trials}"
            )

        stop = False

        while temperature > min_temperature and not stop:
            temperature_iter += 1

            accepted_moves = 0
            uphill_accepts = 0
            skipped_capacity = 0

            for _ in range(self.nb_trials):
                if deadline is not None and time_func() >= deadline:
                    stop = True
                    break

                item_id = rnd_randrange(m_items)
                current_value = cache.packing[item_id]

                # Si el ítem no estaba tomado y no cabe, se descarta antes de evaluar.
                if (
                    current_value == 0
                    and (cache.g_weight + items[item_id].weight) > inst.capacity
                ):
                    skipped_capacity += 1
                    continue

                candidate_gain, new_value = self._candidate_gain_after_flip(
                    inst,
                    cache,
                    item_id,
                )
                energy_gap = candidate_gain - current_gain

                # SA: acepta siempre mejoras y acepta empeoramientos con probabilidad.
                accept = (
                    energy_gap > 0.0
                    or math_exp(energy_gap / temperature) > rnd_random()
                )

                if not accept:
                    continue

                incremental_bitflip(inst, cache, item_id, new_value)
                current_gain = cache.G_gain
                accepted_moves += 1

                if energy_gap <= 0.0:
                    uphill_accepts += 1

            if current_gain > best_gain:
                best_gain = current_gain
                best_packing = cache.packing[:]

            if verbose:
                log_fn(
                    f"[KRP] T_iter={temperature_iter} T={temperature:.4f} "
                    f"trials={self.nb_trials} accepted={accepted_moves} "
                    f"uphill={uphill_accepts} skipped_cap={skipped_capacity} "
                    f"best_gain={best_gain:.4f}"
                )

            if stop:
                break

            temperature *= cooling_factor

        # Reconstruye caché final desde el mejor packing encontrado.
        cache = recompute_history_full(inst, tour, best_packing)

        return cache.tour[:], cache.packing[:]