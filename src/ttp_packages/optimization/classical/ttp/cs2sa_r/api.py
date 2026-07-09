#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/ttp/cs2sa_r/api.py

"""API principal del solver CS2SA-R para Traveling Thief Problem.

Este módulo coordina las fases principales del solver:
    1. Construcción inicial.
    2. Optimización del tour con TSKP.
    3. Optimización del packing con KRP.
    4. Reinicio completo o ligero cuando no hay mejora.
    5. Construcción y verificación de la mejor solución encontrada.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Iterable
from typing import Any, Optional

from src.ttp_packages.domain.solution import TTPSolution
from src.ttp_packages.infrastructure.logging import setup_logger

from .config import (
    DEFAULT_VERBOSE_SECTIONS,
    NO_IMPROVE_PATIENCE,
    RESTART_MODE_FULL,
    RESTART_MODE_LIGHT,
    RESTART_NOISE,
    SEPARATOR,
    START_CITY,
    VERBOSE_ALIAS_ALL,
    VERBOSE_ALIAS_NONE,
    VERBOSE_ALIAS_STEPS,
    VERBOSE_SECTION_CYCLES,
    VERBOSE_SECTION_INITIAL,
    VERBOSE_SECTION_INTEGRITY,
    VERBOSE_SECTION_KRP,
    VERBOSE_SECTION_TSKP,
    VERBOSE_SECTIONS_ALL,
    VERBOSE_SECTIONS_STEPS,
)
from .delta_eval import (
    incremental_bitflip,
    precompute_remaining_distances,
    recompute_history_full,
)
from .initializer import Initializer
from .krp_optimizer import KRPOptimizer
from .tskp_optimizer import TSKPOptimizer


logger = setup_logger(__name__)


class CS2SARSolver:
    """Solver CS2SA-R para el Traveling Thief Problem.

    Attributes:
        inst: Instancia TTP a resolver.
        global_seed: Semilla usada por el solver.
        rnd: Generador pseudoaleatorio compartido.
        initializer: Constructor de soluciones iniciales y reinicios.
        tskp: Optimizador del tour.
        krp: Optimizador del packing.
        restart_noise: Fracción usada para perturbar el packing en reinicio ligero.
        restart_mode: Modo de reinicio.
        no_improve_patience: Cantidad de ciclos sin mejora antes de reiniciar.
    """

    def __init__(
        self,
        inst: Any,
        initializer: Optional[Initializer] = None,
        tskp: Optional[TSKPOptimizer] = None,
        krp: Optional[KRPOptimizer] = None,
        restart_noise: float = RESTART_NOISE,
        restart_mode: str = RESTART_MODE_FULL,
        no_improve_patience: int = NO_IMPROVE_PATIENCE,
        start_city: int = START_CITY,
        seed: Optional[int] = None,
    ):
        """Inicializa el solver CS2SA-R.

        Args:
            inst: Instancia del problema TTP a resolver.
            initializer: Constructor de soluciones iniciales y reinicios.
            tskp: Optimizador TSKP. Si es ``None``, se crea uno por defecto.
            krp: Optimizador KRP. Si es ``None``, se crea uno por defecto.
            restart_noise: Nivel de ruido para el reinicio ligero.
            restart_mode: Modo de reinicio. Puede ser ``"full"`` o ``"light"``.
            no_improve_patience: Ciclos sin mejora antes de reiniciar.
            start_city: Ciudad inicial usada por el constructor de tours.
            seed: Semilla global para reproducibilidad. Si es ``None``, se genera
                una semilla desde el sistema.
        """
        self.inst = inst

        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little") ^ int(time.time_ns())

        self.global_seed = seed
        self.rnd = random.Random(seed)

        self.initializer = initializer or Initializer(
            rng=self.rnd,
            start_city=start_city,
        )

        if hasattr(self.initializer, "set_rng"):
            self.initializer.set_rng(self.rnd)
        else:
            self.initializer.rnd = self.rnd

        self.tskp = tskp or TSKPOptimizer()
        self.krp = krp or KRPOptimizer(rnd_seed=self.rnd.randrange(2**31 - 1))

        self.restart_noise = restart_noise
        self.restart_mode = restart_mode
        self.no_improve_patience = max(1, no_improve_patience)

    @staticmethod
    def _is_section_enabled(verbose_sections: set[str], section: str) -> bool:
        """Indica si una sección de logging está habilitada.

        Args:
            verbose_sections: Conjunto de secciones activas.
            section: Sección consultada.

        Returns:
            ``True`` si la sección está activa.
        """
        return section in verbose_sections

    def _normalize_verbose_sections(
        self,
        verbose_sections: Optional[Iterable[str] | str] = None,
    ) -> set[str]:
        """Normaliza la configuración de logging por secciones.

        Args:
            verbose_sections: Sección, alias o colección de secciones. Acepta
                ``"all"``, ``"steps"``, ``"none"`` o nombres de secciones
                individuales.

        Returns:
            Conjunto de secciones habilitadas.

        Raises:
            ValueError: Si se recibe una sección o alias desconocido.
        """
        valid_sections = set(VERBOSE_SECTIONS_ALL)
        valid_aliases = {
            VERBOSE_ALIAS_ALL,
            VERBOSE_ALIAS_STEPS,
            VERBOSE_ALIAS_NONE,
        }

        if verbose_sections is None:
            return set(DEFAULT_VERBOSE_SECTIONS)

        if isinstance(verbose_sections, str):
            requested = {verbose_sections}
        else:
            requested = set(verbose_sections)

        if VERBOSE_ALIAS_NONE in requested:
            return set()

        normalized: set[str] = set()

        if VERBOSE_ALIAS_ALL in requested:
            normalized.update(VERBOSE_SECTIONS_ALL)

        if VERBOSE_ALIAS_STEPS in requested:
            normalized.update(VERBOSE_SECTIONS_STEPS)

        for token in requested:
            if token in valid_sections:
                normalized.add(token)
            elif token not in valid_aliases:
                raise ValueError(
                    f"Sección de verbose desconocida: {token}. "
                    f"Válidas: {sorted(valid_sections | valid_aliases)}"
                )

        return normalized

    def _restart_packing(self, tour: list[int], packing: list[int]) -> list[int]:
        """Aplica un reinicio ligero sobre el packing.

        El reinicio ligero reconstruye un packing candidato a partir de scores
        aproximados por ítem y luego aplica flips aleatorios controlados por
        ``restart_noise``.

        Args:
            tour: Tour actual.
            packing: Packing actual asociado al tour.

        Returns:
            Nuevo packing perturbado.
        """
        inst = self.inst
        items = inst.items
        capacity = inst.capacity
        rent_per_time = inst.rent_per_time
        max_speed = inst.max_speed
        min_speed = inst.min_speed

        speed_diff = max_speed - min_speed
        speed_coef = speed_diff / capacity if capacity > 0 else 0.0

        # Se parte desde packing vacío para estimar la ganancia individual de ítems.
        cache = recompute_history_full(inst, tour, [0] * inst.m_items)
        remaining_distances = precompute_remaining_distances(inst, tour)
        pos_in_tour = cache.pos_in_tour
        accumulated_weight = cache.w_acc

        candidates: list[int] = []
        item_scores: dict[int, float] = {}

        m_items = inst.m_items
        for item_id in range(m_items):
            item = items[item_id]
            item_weight = item.weight

            if item_weight <= 0:
                continue

            candidates.append(item_id)

            item_position = pos_in_tour[item.city_id]
            base_weight = accumulated_weight[item_position]

            new_weight = base_weight + item_weight
            new_speed = max_speed - (new_weight * speed_coef)
            if new_speed < min_speed:
                new_speed = min_speed

            old_speed = max_speed - (base_weight * speed_coef)
            if old_speed < min_speed:
                old_speed = min_speed

            delta_inv_speed = (1.0 / new_speed) - (1.0 / old_speed)
            score = (
                item.profit
                - rent_per_time
                * remaining_distances[item_position]
                * delta_inv_speed
            )
            item_scores[item_id] = score

        candidates.sort(key=item_scores.get, reverse=True)

        current_weight = 0.0
        new_packing = [0] * m_items

        for item_id in candidates:
            if item_scores[item_id] <= 0:
                continue

            item_weight = items[item_id].weight
            if current_weight + item_weight <= capacity:
                new_packing[item_id] = 1
                current_weight += item_weight

        cache = recompute_history_full(inst, tour, new_packing)

        n_flips = max(1, int(self.restart_noise * m_items))
        rnd_randrange = self.rnd.randrange

        for _ in range(n_flips):
            item_id = rnd_randrange(m_items)
            old_value = cache.packing[item_id]
            new_value = 1 - old_value

            item_weight = items[item_id].weight
            if new_value == 1 and (cache.g_weight + item_weight) > capacity:
                continue

            previous_gain = cache.G_gain
            previous_profit = cache.g_profit
            previous_time = cache.f_time
            previous_weight = cache.g_weight

            incremental_bitflip(inst, cache, item_id, new_value)

            # Rechaza perturbaciones que deterioran demasiado el objetivo.
            if (cache.G_gain - previous_gain) < -1.0:
                incremental_bitflip(inst, cache, item_id, old_value)

                cache.G_gain = previous_gain
                cache.g_profit = previous_profit
                cache.f_time = previous_time
                cache.g_weight = previous_weight

        return cache.packing[:]

    def _restart_full(
        self,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[int], list[int]]:
        """Aplica un reinicio completo construyendo una solución nueva.

        Args:
            verbose: Si es True, habilita logs del constructor.
            log_fn: Función usada para emitir logs.

        Returns:
            Tupla ``(tour, packing)`` generada por el initializer.
        """
        tour, packing = self.initializer.build_restart(
            self.inst,
            verbose=verbose,
            log_fn=log_fn,
        )
        return tour, packing

    def verify_value_integrity(
        self,
        res: dict,
        solution: TTPSolution,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Verifica consistencia entre métricas crudas y solución TTP.

        Args:
            res: Diccionario con ``time``, ``profit`` y ``gain`` calculados por
                la caché del solver.
            solution: Solución TTP final.
            verbose: Si es True, muestra el detalle de la comparación.
            log_fn: Función usada para emitir logs.

        Returns:
            ``True`` si las métricas coinciden exactamente.
        """
        if log_fn is None:
            log_fn = logger.info

        time_raw = float(res["time"])
        time_obj = float(solution.time)
        time_match = time_raw == time_obj

        profit_raw = float(res["profit"])
        profit_obj = float(solution.profit)
        profit_match = profit_raw == profit_obj

        gain_raw = float(res["gain"])
        gain_obj = float(solution.objective)
        gain_match = gain_raw == gain_obj

        ok = time_match and profit_match and gain_match

        if verbose:
            log_fn(f"\n{SEPARATOR}")
            log_fn("VERIFICACIÓN DE INTEGRIDAD NUMÉRICA")
            log_fn(SEPARATOR)

            if time_match:
                log_fn(f"✅ Time: Coinciden correctamente. ({time_obj})")
            if profit_match:
                log_fn(f"✅ Profit: Coinciden correctamente. ({profit_obj})")
            if gain_match:
                log_fn(f"✅ Gain: Coinciden correctamente. ({gain_obj})")

        if not time_match:
            logger.warning(
                "❌ Time: NO coinciden. res: %s vs sol: %s",
                time_raw,
                time_obj,
            )
        if not profit_match:
            logger.warning(
                "❌ Profit: NO coinciden. res: %s vs sol: %s",
                profit_raw,
                profit_obj,
            )
        if not gain_match:
            logger.warning(
                "❌ Gain: NO coinciden. res: %s vs sol: %s",
                gain_raw,
                gain_obj,
            )

        if verbose:
            log_fn(SEPARATOR)

        return ok

    def verify_solution_integrity(
        self,
        res: dict,
        solution: TTPSolution,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Verifica consistencia entre arreglos crudos y solución TTP.

        Args:
            res: Diccionario con ``tour`` y ``packing`` calculados por el solver.
            solution: Solución TTP final.
            verbose: Si es True, muestra el detalle de la comparación.
            log_fn: Función usada para emitir logs.

        Returns:
            ``True`` si tour y packing coinciden exactamente.
        """
        if log_fn is None:
            log_fn = logger.info

        tour_raw = list(res["tour"])
        tour_obj = list(solution.tour)
        tour_match = tour_raw == tour_obj

        packing_raw = list(res["packing"])
        packing_obj = list(solution.packing)
        packing_match = packing_raw == packing_obj

        ok = tour_match and packing_match

        if verbose:
            log_fn(f"\n{SEPARATOR}")
            log_fn("VERIFICACIÓN DE INTEGRIDAD ESTRUCTURAL")
            log_fn(SEPARATOR)

            if tour_match:
                log_fn("✅ Tour: Coinciden correctamente.")
            if packing_match:
                log_fn("✅ Packing/Picking: Coinciden correctamente.")

        if not tour_match:
            logger.warning(
                "❌ Tour: NO coinciden. Largo res: %s vs Largo sol: %s",
                len(tour_raw),
                len(tour_obj),
            )
        if not packing_match:
            logger.warning(
                "❌ Packing: NO coinciden. Largo res: %s vs Largo sol: %s",
                len(packing_raw),
                len(packing_obj),
            )

        if verbose:
            log_fn(SEPARATOR)

        return ok

    def solve(
        self,
        time_budget_s: float = 60.0,
        verbose_sections: Optional[Iterable[str] | str] = None,
        verify_integrity: bool = True,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> TTPSolution:
        """Ejecuta el ciclo principal de CS2SA-R.

        En cada ciclo se optimiza primero el tour con TSKP y luego el packing con
        KRP. Si el intento actual no mejora durante ``no_improve_patience``
        ciclos, se aplica un reinicio completo o ligero.

        Args:
            time_budget_s: Presupuesto total de tiempo en segundos.
            verbose_sections: Secciones de logging habilitadas. Acepta nombres
                individuales, ``"steps"``, ``"all"`` o ``"none"``.
            verify_integrity: Si es True, ejecuta verificaciones de integridad al
                final.
            log_fn: Función usada para emitir logs.

        Returns:
            Mejor solución encontrada dentro del presupuesto de tiempo.
        """
        if log_fn is None:
            log_fn = logger.info

        enabled_sections = self._normalize_verbose_sections(
            verbose_sections=verbose_sections,
        )

        verbose_cycles = self._is_section_enabled(
            enabled_sections,
            VERBOSE_SECTION_CYCLES,
        )
        verbose_initial = self._is_section_enabled(
            enabled_sections,
            VERBOSE_SECTION_INITIAL,
        )
        verbose_tskp = self._is_section_enabled(
            enabled_sections,
            VERBOSE_SECTION_TSKP,
        )
        verbose_krp = self._is_section_enabled(
            enabled_sections,
            VERBOSE_SECTION_KRP,
        )
        verbose_integrity = self._is_section_enabled(
            enabled_sections,
            VERBOSE_SECTION_INTEGRITY,
        )

        time_func = time.time
        deadline = time_func() + time_budget_s

        inst_ref = self.inst
        tskp_opt = self.tskp
        krp_opt = self.krp

        # Construcción inicial: entrega el primer tour y packing factible.
        current_tour, current_packing = self.initializer.build_initial(
            inst_ref,
            verbose=verbose_initial,
            log_fn=log_fn,
        )
        current_cache = recompute_history_full(
            inst_ref,
            current_tour,
            current_packing,
        )

        # Mejor global: solución que se devolverá al final.
        best_global_gain = current_cache.G_gain
        best_tour = current_tour[:]
        best_pack = current_packing[:]

        # Mejor del intento actual: se reinicia después de cada restart.
        best_run_gain = current_cache.G_gain

        idle_steps = 0
        cycles = 0
        restarts_done = 0

        while time_func() < deadline:
            cycles += 1

            # TSKP modifica el tour manteniendo el packing actual.
            current_tour, current_packing = tskp_opt.optimize(
                inst_ref,
                current_tour,
                current_packing,
                max_passes=None,
                verbose=verbose_tskp,
                log_fn=log_fn,
                deadline=deadline,
            )

            # KRP modifica el packing manteniendo fijo el tour actual.
            current_tour, current_packing = krp_opt.optimize(
                inst_ref,
                current_tour,
                current_packing,
                verbose=verbose_krp,
                log_fn=log_fn,
                deadline=deadline,
            )

            current_cache = recompute_history_full(
                inst_ref,
                current_tour,
                current_packing,
            )

            # El ciclo completo TSKP + KRP se compara como una única propuesta.
            current_gain = current_cache.G_gain

            if current_gain > best_global_gain + 1e-9:
                best_global_gain = current_gain
                best_tour = current_tour[:]
                best_pack = current_packing[:]

            if current_gain > best_run_gain + 1e-9:
                best_run_gain = current_gain
                idle_steps = 0
            else:
                idle_steps += 1

            # Cuando el intento actual se estanca, se reinicia la búsqueda.
            if idle_steps >= self.no_improve_patience and time_func() < deadline:
                if verbose_cycles:
                    log_fn(
                        f"[CS2SAR] ===> RESTART (cycle={cycles}, "
                        f"run_best={best_run_gain:.4f}, "
                        f"global_best={best_global_gain:.4f})"
                    )

                if self.restart_mode == RESTART_MODE_FULL:
                    current_tour, current_packing = self._restart_full(
                        verbose=verbose_initial,
                        log_fn=log_fn,
                    )
                else:
                    current_packing = self._restart_packing(
                        current_tour,
                        current_packing,
                    )

                current_cache = recompute_history_full(
                    inst_ref,
                    current_tour,
                    current_packing,
                )

                best_run_gain = current_cache.G_gain
                idle_steps = 0
                restarts_done += 1

            if verbose_cycles:
                log_fn(
                    f"[CS2SAR] cycle={cycles} | "
                    f"current={current_cache.G_gain:.4f} "
                    f"run_best={best_run_gain:.4f} "
                    f"global_best={best_global_gain:.4f} "
                    f"| idle={idle_steps} restart={restarts_done}"
                )

        best_cache = recompute_history_full(inst_ref, best_tour, best_pack)
        solution = TTPSolution(inst_ref, best_tour[:], best_pack[:])

        # Ejecuta la validación interna de TTPSolution si check está implementado como propiedad.
        solution.check

        if verify_integrity:
            res = {
                "gain": best_cache.G_gain,
                "time": best_cache.f_time,
                "profit": best_cache.g_profit,
                "tour": best_tour[:],
                "packing": best_pack[:],
            }

            self.verify_solution_integrity(
                res,
                solution,
                verbose=verbose_integrity,
                log_fn=log_fn,
            )
            self.verify_value_integrity(
                res,
                solution,
                verbose=verbose_integrity,
                log_fn=log_fn,
            )

        return solution