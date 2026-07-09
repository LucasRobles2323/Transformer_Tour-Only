#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/domain/solution.py

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.ttp_packages.infrastructure.logging import setup_logger
    
from .instance import TTPInstance
from .constants import DEBUG_LINE_WIDTH
from .objective import calculate_objective_function


# Inicialización del logger
logger = setup_logger(__name__)

class TTPSolution:
    """Representa una solución TTP y sus métricas evaluadas.

    Attributes:
        inst: Instancia TTP asociada.
        tour: Secuencia de ciudades de la solución.
        packing: Vector binario de selección de ítems.
        profit: Beneficio total calculado.
        time: Tiempo total calculado.
        objective: Valor objetivo calculado.
    """

    def __init__(
        self,
        inst: TTPInstance,
        tour: Optional[List[int]] = None,
        packing_plan: Optional[List[int]] = None
    ):
        """Inicializa una solución TTP.

        Args:
            inst: Instancia TTP asociada.
            tour: Tour inicial. Si se entrega junto con ``packing_plan``, se evalúa
                inmediatamente.
            packing_plan: Vector binario inicial de selección de ítems.
        """
        self.inst = inst
        self.tour = tour
        self.packing = packing_plan
        self.profit, self.time, self.objective = 0.0, 0.0, 0.0

        if self.tour is not None and self.packing is not None:
            self.compute_benefit()
    
    def check(self, start_city: int = 0) -> bool:
        """Valida que la solución sea estructuralmente factible.

        La validación revisa que exista instancia, tour y packing; que el packing
        sea binario; que el tour visite cada ciudad exactamente una vez; y que el
        peso acumulado no exceda la capacidad.

        Args:
            start_city: Ciudad inicial esperada para el tour.

        Returns:
            ``True`` si la solución es válida; ``False`` en caso contrario.
        """
        if self.inst is None:
            logger.warning("Solución inválida: inst es None.")
            return False

        if self.tour is None:
            logger.warning("Solución inválida: tour es None.")
            return False

        if self.packing is None:
            logger.warning("Solución inválida: packing es None.")
            return False

        n = int(self.inst.n_cities)
        m = int(self.inst.m_items)
        cap = float(self.inst.capacity)

        packing_arr = np.asarray(self.packing, dtype=np.int64)

        if packing_arr.size != m:
            logger.warning(
                "Packing inválido: len(packing)=%s != m_items=%s.",
                packing_arr.size,
                m,
            )
            return False

        if not np.isin(packing_arr, (0, 1)).all():
            invalid_positions = np.flatnonzero(~np.isin(packing_arr, (0, 1)))
            first_invalid = int(invalid_positions[0])
            logger.warning(
                "Packing inválido: packing[%s]=%s no es binario.",
                first_invalid,
                packing_arr[first_invalid],
            )
            return False

        tour = list(map(int, self.tour))

        if not tour:
            logger.warning("Tour inválido: está vacío.")
            return False

        # Se acepta un ciclo cerrado explícito, por ejemplo [0, 1, 2, 0].
        if len(tour) == n + 1 and tour[0] == tour[-1]:
            tour = tour[:-1]

        if len(tour) != n:
            logger.warning("Tour inválido: len(tour)=%s != n_cities=%s.", len(tour), n)
            return False

        if tour[0] != start_city:
            logger.warning(
                "Tour inválido: comienza en %s y no en start_city=%s.",
                tour[0],
                start_city,
            )
            return False

        expected_cities = set(range(n))
        received_cities = set(tour)

        if len(received_cities) != n:
            logger.warning("Tour inválido: hay ciudades repetidas.")
            return False

        if received_cities != expected_cities:
            logger.warning("Tour inválido: no recorre exactamente las ciudades 0..N-1.")
            return False

        current_weight = 0.0

        for city_id in tour:
            for item_id in self.inst.cities[city_id].items:
                if packing_arr[item_id]:
                    current_weight += float(self.inst.items[item_id].weight)

            # La capacidad debe revisarse después de cada ciudad porque el TTP usa
            # peso acumulado durante el recorrido.
            if current_weight > cap:
                logger.warning(
                    "Solución inválida: capacidad excedida en city %s. "
                    "Peso actual=%.4f > capacidad=%.4f.",
                    city_id,
                    current_weight,
                    cap,
                )
                return False

        return True

    def compute_benefit(self):
        """Recalcula profit, tiempo y función objetivo de la solución actual."""
        if self.inst is None or self.tour is None or self.packing is None:
            return

        obj_val, time_val, profit_val = calculate_objective_function(
            self.inst, self.tour, self.packing
        )
        
        self.profit = profit_val
        self.time = time_val
        self.objective = obj_val

    def update_solution(self, tour: List[int], packing: List[int]):
        """Actualiza la solución y recalcula sus métricas.

        Args:
            tour: Nuevo tour.
            packing: Nuevo vector binario de selección de ítems.
        """
        self.tour = tour
        self.packing = packing
        self.compute_benefit()

    def print_sol(self):
        """Registra un resumen legible de la solución actual."""
        items_count = int(np.asarray(self.packing).sum()) if self.packing is not None else 0
        cost = self.inst.rent_per_time * self.time
        
        tour_str = f"{self.tour[:10]}..." if self.tour and len(self.tour) > 10 else f"{self.tour}"
        
        msg = [
            "\n" + "=" * DEBUG_LINE_WIDTH,
            " RESUMEN DE SOLUCIÓN TTP",
            "-" * DEBUG_LINE_WIDTH,
            f"Tour (primeros 10): {tour_str}",
            f"Items recogidos: {items_count}",
            f"Tiempo (Time/f)     : {self.time:.2f}",
            f"Costo Renta (R)     : {self.inst.rent_per_time:.2f}",
            f"Ganancia (Profit/g) : {self.profit:.2f}",
            f"Costo (f*R)         : {cost:.2f}",
            f"G = {self.profit:.2f} - ({self.inst.rent_per_time:.2f} * {self.time:.2f})",
            f"FINAL OBJETIVO      : {self.objective:.2f}",
            "=" * DEBUG_LINE_WIDTH
        ]
        logger.info("\n".join(msg))