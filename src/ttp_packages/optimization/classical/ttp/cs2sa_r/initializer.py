#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/ttp/cs2sa_r/initializer.py

"""Construcción de soluciones iniciales y reinicios para CS2SA-R.

Este módulo construye el tour inicial con ``lk_heuristic`` y genera el packing
inicial mediante criterios heurísticos basados en ``t1``, ``t2`` y ``t3``.

También incluye el boost por bit-flips para instancias pequeñas y la generación
de packings aleatorios para reinicios.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from src.ttp_packages.infrastructure.logging import setup_logger

from .config import (
    INIT_BOOST_CITY_CUTOFF,
    INIT_ELIMINATION_CITY_CUTOFF,
    LK_DEFAULT_LOGGING_LEVEL,
    LK_DEFAULT_SOLUTION_METHOD,
    START_CITY,
    TSPLIB_EOF,
    TSPLIB_NODE_COORD_SECTION,
    TSPLIB_TEMP_INPUT_FILENAME,
    TSPLIB_TEMP_OUTPUT_FILENAME,
)
from .delta_eval import incremental_bitflip, recompute_history_full


logger = setup_logger(__name__)


class Initializer:
    """Construye soluciones iniciales para el solver CS2SA-R.

    El tour inicial se obtiene resolviendo el subproblema TSP con ``lk_heuristic``.
    Luego se construye el packing inicial con la heurística de inserción,
    eliminación y boost del solver.

    Attributes:
        start_city: Ciudad inicial canónica del tour.
        lk_solution_method: Método de ``lk_heuristic`` usado para construir el
            tour inicial.
        lk_backtracking: Parámetros de backtracking para ``lk_heuristic``.
        lk_reduction_level: Nivel de reducción configurado para ``lk_heuristic``.
        lk_reduction_cycle: Ciclo de reducción configurado para ``lk_heuristic``.
        rnd: Generador pseudoaleatorio compartido con el solver.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
        start_city: int = START_CITY,
        lk_solution_method: str = LK_DEFAULT_SOLUTION_METHOD,
        lk_backtracking: tuple[int, int] = (5, 5),
        lk_reduction_level: int = 4,
        lk_reduction_cycle: int = 4,
    ):
        """Inicializa el constructor de soluciones iniciales.

        Args:
            rng: Generador aleatorio compartido por el solver.
            start_city: Ciudad inicial canónica del tour.
            lk_solution_method: Método de ``lk_heuristic`` usado para construir
                el tour TSP inicial.
            lk_backtracking: Parámetros de backtracking para ``lk_heuristic``.
            lk_reduction_level: Nivel de reducción para ``lk_heuristic``.
            lk_reduction_cycle: Ciclo de reducción para ``lk_heuristic``.
        """
        self.start_city = int(start_city)

        self.lk_solution_method = lk_solution_method
        self.lk_backtracking = lk_backtracking
        self.lk_reduction_level = int(lk_reduction_level)
        self.lk_reduction_cycle = int(lk_reduction_cycle)

        if rng is not None:
            self.rnd = rng
        elif not hasattr(self, "rnd"):
            self.rnd = random.Random()

    def set_rng(self, rng: random.Random) -> None:
        """Inyecta el generador aleatorio del solver.

        Args:
            rng: Instancia de ``random.Random`` compartida por el solver.
        """
        self.rnd = rng

    def _rotate_to_start(self, tour: list[int], start: int) -> list[int]:
        """Rota un tour hamiltoniano para que comience en ``start``.

        No modifica el orden cíclico del recorrido; solo cambia el punto desde
        el que se empieza a leer la permutación.

        Args:
            tour: Secuencia de ciudades que representa un tour hamiltoniano.
            start: Ciudad que debe quedar en la primera posición.

        Returns:
            Tour rotado para que ``start`` quede en la posición ``0``.

        Raises:
            ValueError: Si ``tour`` está vacío.
            ValueError: Si ``start`` no pertenece a ``tour``.
        """
        if not tour:
            raise ValueError("LK devolvió un tour vacío")

        if start not in tour:
            raise ValueError(f"El tour LK no contiene start_city={start}")

        pos = tour.index(start)
        return tour[pos:] + tour[:pos]

    def _compute_lk_runs(self, n_cities: int) -> int:
        """Calcula la cantidad de ejecuciones de ``lk_heuristic`` según el tamaño.

        Args:
            n_cities: Número de ciudades de la instancia.

        Returns:
            Cantidad de ejecuciones ``runs`` para construir el tour inicial.
        """
        runs = 1

        if n_cities > 100:
            runs = 10
        if n_cities > 1000:
            runs = 80
        if n_cities > 10000:
            runs = 500
        if n_cities > 30000:
            runs = 800

        return runs

    def _write_tsplib_euc2d(self, inst, tsp_path: str) -> None:
        """Exporta una instancia TTP al formato TSPLIB EUC_2D.

        Solo se exporta el subproblema TSP de la instancia: ciudades y
        coordenadas. Los ítems no se escriben porque ``lk_heuristic`` resuelve
        únicamente el tour.

        Args:
            inst: Instancia TTP de entrada.
            tsp_path: Ruta del archivo temporal ``.tsp`` a escribir.
        """
        with open(tsp_path, "w", encoding="utf-8") as file:
            file.write(f"NAME : {getattr(inst, 'name', 'inst')}\n")
            file.write("TYPE: TSP\n")
            file.write(f"DIMENSION : {inst.n_cities}\n")
            file.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
            file.write("NODE_COORD_SECTION\n")

            for i, city in enumerate(inst.cities, start=1):
                file.write(f"{i} {float(city.x)} {float(city.y)}\n")

            file.write("EOF\n")

    def _parse_lk_solution_tour(self, sol_path: str, inst) -> list[int]:
        """Reconstruye el tour exportado por ``lk_heuristic`` usando coordenadas.

        La salida ``.tsp`` de ``lk_heuristic`` reordena las filas del
        ``NODE_COORD_SECTION`` según el tour encontrado, pero la primera columna
        no preserva necesariamente los identificadores originales de ciudad. Por
        eso, el tour se reconstruye emparejando coordenadas de salida con
        coordenadas originales de la instancia.

        Si varias ciudades comparten la misma coordenada, se mantiene una cola de
        ``city_id`` por coordenada y se consume una ciudad por cada aparición.

        Args:
            sol_path: Ruta del archivo ``.tsp`` exportado por ``lk_heuristic``.
            inst: Instancia TTP original.

        Returns:
            Lista de ``city_id`` base cero en el orden del tour reconstruido.

        Raises:
            RuntimeError: Si una coordenada no puede mapearse a una ciudad
                original o si quedan ciudades sin consumir.
        """
        order: list[int] = []
        in_node_section = False

        # Mapa coordenada -> cola de city_id originales. La cola maneja empates.
        coord_to_city_ids: dict[tuple[float, float], deque[int]] = defaultdict(deque)

        for city_id, city in enumerate(inst.cities):
            key = (float(city.x), float(city.y))
            coord_to_city_ids[key].append(city_id)

        with open(sol_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()

                if not line:
                    continue

                upper = line.upper()

                if upper.startswith(TSPLIB_NODE_COORD_SECTION):
                    in_node_section = True
                    continue

                if upper == TSPLIB_EOF:
                    break

                if not in_node_section:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                # La primera columna no preserva city_id; el orden útil está en
                # el reordenamiento de filas del NODE_COORD_SECTION.
                x = float(parts[1])
                y = float(parts[2])
                key = (x, y)

                if key not in coord_to_city_ids or not coord_to_city_ids[key]:
                    raise RuntimeError(
                        f"No encontré una ciudad disponible para coordenada {key} "
                        f"en archivo de salida {sol_path}"
                    )

                order.append(coord_to_city_ids[key].popleft())

        leftovers = sum(len(queue) for queue in coord_to_city_ids.values())

        if leftovers != 0:
            raise RuntimeError(
                f"Tour LK incompleto o inconsistente: quedaron {leftovers} "
                "ciudades sin mapear tras parsear la salida."
            )

        return order

    def _lk_heuristic_tour(
        self,
        inst,
        start: int = 0,
        verbose: bool = False,
        log_fn: Callable[[str], None] | None = None,
    ) -> list[int]:
        """Construye el tour TSP inicial usando ``lk_heuristic``.

        Args:
            inst: Instancia TTP de entrada.
            start: Ciudad desde la que debe comenzar el tour canónico.
            verbose: Si es True, muestra logs de inicialización.
            log_fn: Función opcional de logging.

        Returns:
            Lista de identificadores de ciudad en el orden del tour.

        Raises:
            RuntimeError: Si ``lk_heuristic`` no genera una salida válida.
        """
        # Import local: lk_heuristic solo se necesita para construir tours LK.
        import lk_heuristic
        from lk_heuristic.utils.solver_funcs import solve as lk_solve

        if log_fn is None:
            log_fn = logger.info

        # Aumenta la cantidad de ejecuciones LK en instancias más grandes.
        nb_runs = self._compute_lk_runs(inst.n_cities)

        if verbose:
            log_fn(
                f"[INIT] LK start n={inst.n_cities} runs={nb_runs} "
                f"method={self.lk_solution_method}"
            )

        # lk_heuristic trabaja con archivos TSPLIB; se usa un directorio temporal
        # para aislar archivos de entrada/salida.
        with tempfile.TemporaryDirectory() as tmpd:
            tsp_path = os.path.join(tmpd, TSPLIB_TEMP_INPUT_FILENAME)

            # Stem único para evitar colisiones si hay ejecuciones paralelas.
            output_stem = f"{Path(TSPLIB_TEMP_OUTPUT_FILENAME).stem}_{uuid4().hex}"

            # Exporta únicamente el subproblema TSP.
            self._write_tsplib_euc2d(inst, tsp_path)

            lk_solve(
                tsp_file=tsp_path,
                solution_method=self.lk_solution_method,
                runs=max(1, nb_runs),
                backtracking=self.lk_backtracking,
                reduction_level=self.lk_reduction_level,
                reduction_cycle=self.lk_reduction_cycle,
                tour_type="cycle",
                file_name=output_stem,
                logging_level=LK_DEFAULT_LOGGING_LEVEL,
            )

            pkg_dir = Path(lk_heuristic.__file__).resolve().parent
            pkg_solutions_dir = pkg_dir / "solutions"

            search_roots = [
                Path(tmpd),
                Path(os.getcwd()),
                Path(os.getcwd()) / "solutions",
                pkg_solutions_dir,
            ]

            candidates = []
            for root in search_roots:
                if root.exists():
                    candidates.extend(root.glob(f"{output_stem}*.tsp"))

            if not candidates:
                raise RuntimeError(
                    f"No encontré archivo de salida de lk_heuristic para "
                    f"stem='{output_stem}'. Busqué en: {search_roots}"
                )

            sol_path = str(max(candidates, key=lambda path: path.stat().st_mtime))

            if not os.path.exists(sol_path):
                raise RuntimeError(f"No se generó el archivo esperado: {sol_path}")

            # Reinterpreta el archivo de salida para recuperar city_id originales.
            order = self._parse_lk_solution_tour(sol_path, inst)

        expected = set(range(inst.n_cities))
        found = set(order)

        if len(order) != inst.n_cities:
            raise RuntimeError(
                f"Tour inválido: esperaba {inst.n_cities} ciudades "
                f"y obtuvo {len(order)}"
            )

        if len(found) != inst.n_cities or found != expected:
            raise RuntimeError("Tour LK inválido: hay duplicados o faltan ciudades")

        # Normaliza el ciclo para que siempre empiece en la ciudad canónica.
        rotated = self._rotate_to_start(order, start)

        if verbose:
            log_fn(f"[INIT] LK done len={len(rotated)} start_city={start}")

        return rotated

    def _zeros_packing(self, inst) -> list[int]:
        """Crea un packing vacío para la instancia.

        Args:
            inst: Instancia TTP de entrada.

        Returns:
            Lista binaria de longitud ``m_items`` con todos los valores en cero.
        """
        return [0] * inst.m_items

    def _speed_from_weight(self, inst, weight: float) -> float:
        """Calcula la velocidad efectiva para un peso acumulado.

        Args:
            inst: Instancia TTP de entrada.
            weight: Peso acumulado en la mochila.

        Returns:
            Velocidad efectiva acotada inferiormente por ``min_speed``.
        """
        if inst.capacity <= 0:
            return inst.max_speed

        speed = inst.max_speed - (
            ((inst.max_speed - inst.min_speed) / float(inst.capacity)) * weight
        )

        return max(inst.min_speed, speed)

    def _distance_to_end(self, inst, tour: list[int]) -> list[float]:
        """Calcula la distancia restante desde cada posición del tour.

        Args:
            inst: Instancia TTP de entrada.
            tour: Tour de ciudades.

        Returns:
            Lista donde cada posición contiene la distancia restante hasta completar
            el ciclo.
        """
        if inst.distance_matrix is None:
            inst.create_distance_matrix()

        distance_matrix = inst.distance_matrix
        if distance_matrix is None:
            raise RuntimeError("No se pudo crear la matriz de distancias.")

        n_positions = len(tour)
        remaining_distance = [0.0] * n_positions

        if n_positions == 0:
            return remaining_distance

        # Incluye el arco de cierre última ciudad -> primera ciudad.
        remaining_distance[n_positions - 1] = float(
            distance_matrix[tour[n_positions - 1], tour[0]]
        )

        # Avanza hacia atrás acumulando distancia hasta cerrar el ciclo.
        for position in range(n_positions - 2, -1, -1):
            remaining_distance[position] = (
                remaining_distance[position + 1]
                + float(distance_matrix[tour[position + 1], tour[position]])
            )

        return remaining_distance

    def _sorted_items_by_t1_score(
        self,
        inst,
        tour: list[int],
    ) -> tuple[list[int], list[float], dict]:
        """Ordena ítems según el score inicial de la heurística.

        Args:
            inst: Instancia TTP de entrada.
            tour: Tour actual de ciudades.

        Returns:
            Tupla con ítems ordenados, distancias restantes y mapa
            ``city_id -> posición``.
        """
        # Distancia restante desde cada posición hasta cerrar el ciclo.
        remaining_distance = self._distance_to_end(inst, tour)

        # Permite ubicar rápidamente la posición donde se recoge cada ítem.
        pos_in_tour = {city_id: idx for idx, city_id in enumerate(tour)}

        # Constantes del modelo de velocidad lineal y del costo temporal.
        speed_coef = (
            (inst.max_speed - inst.min_speed) / float(inst.capacity)
            if inst.capacity > 0
            else 0.0
        )
        rent_per_time = inst.rent_per_time
        max_speed = inst.max_speed

        scored = []

        for item_id, item in enumerate(inst.items):
            if item.weight <= 0:
                # Ítems con peso no positivo no participan en esta heurística.
                continue

            # Posición del tour donde se recoge el ítem.
            item_position = pos_in_tour[item.city_id]

            # t1 estima el sobrecoste temporal de transportar solo este ítem.
            t1 = remaining_distance[item_position] * (
                (1.0 / (max_speed - speed_coef * item.weight))
                - (1.0 / max_speed)
            )

            # Score: beneficio neto aproximado dividido por peso.
            score = (item.profit - rent_per_time * t1) / item.weight
            scored.append((score, item_id))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [item_id for _, item_id in scored], remaining_distance, pos_in_tour

    def _insert_t2_packing(self, inst, tour: list[int]) -> tuple[list[int], list[int]]:
        """Construye un packing inicial usando solo el criterio ``t2``.

        Args:
            inst: Instancia TTP de entrada.
            tour: Tour actual de ciudades.

        Returns:
            Tupla ``(packing, inserted_items)``.
        """
        packing = self._zeros_packing(inst)

        sorted_items, remaining_distance, pos_in_tour = self._sorted_items_by_t1_score(
            inst,
            tour,
        )

        speed_coef = (
            (inst.max_speed - inst.min_speed) / float(inst.capacity)
            if inst.capacity > 0
            else 0.0
        )
        rent_per_time = inst.rent_per_time
        max_speed = inst.max_speed
        capacity = float(inst.capacity)

        current_weight = 0.0
        inserted_items: list[int] = []

        for item_id in sorted_items:
            item = inst.items[item_id]
            item_weight = float(item.weight)

            if current_weight + item_weight > capacity:
                continue

            item_position = pos_in_tour[item.city_id]
            t2 = remaining_distance[item_position] * (
                (1.0 / (max_speed - speed_coef * (current_weight + item_weight)))
                - (1.0 / (max_speed - speed_coef * current_weight))
            )

            if item.profit > rent_per_time * t2:
                packing[item_id] = 1
                current_weight += item_weight
                inserted_items.append(item_id)

        return packing, inserted_items

    def _insert_and_eliminate_packing(self, inst, tour: list[int]) -> list[int]:
        """Construye un packing inicial mediante inserción y eliminación.

        Args:
            inst: Instancia TTP de entrada.
            tour: Tour actual de ciudades.

        Returns:
            Packing binario resultante tras inserción y posible eliminación.
        """
        # Comienza con packing vacío y agrega ítems que cumplan criterios t2/t3.
        packing = self._zeros_packing(inst)

        # Orden inicial de exploración de ítems según t1.
        sorted_items, remaining_distance, pos_in_tour = self._sorted_items_by_t1_score(
            inst,
            tour,
        )

        speed_coef = (
            (inst.max_speed - inst.min_speed) / float(inst.capacity)
            if inst.capacity > 0
            else 0.0
        )
        rent_per_time = inst.rent_per_time
        max_speed = inst.max_speed
        capacity = float(inst.capacity)

        current_weight = 0.0
        inserted_items: list[int] = []

        for item_id in sorted_items:
            item = inst.items[item_id]
            item_weight = float(item.weight)

            if current_weight + item_weight > capacity:
                # Si no cabe, se descarta sin evaluar t2/t3.
                continue

            # Posición del tour donde se recoge el ítem.
            item_position = pos_in_tour[item.city_id]

            # t2 estima el incremento de tiempo al añadir el ítem con el peso actual.
            t2 = remaining_distance[item_position] * (
                (1.0 / (max_speed - speed_coef * (current_weight + item_weight)))
                - (1.0 / (max_speed - speed_coef * current_weight))
            )

            accept = item.profit > rent_per_time * t2

            # Si t2 no acepta, t3 evalúa el impacto temporal considerando el peso acumulado.
            if not accept and current_weight > 0.0 and remaining_distance[0] > 0.0:
                weight_per_distance = current_weight / remaining_distance[0]
                speed_with_item = max_speed - speed_coef * (current_weight + item_weight)
                speed_without_item = max_speed - speed_coef * current_weight

                numerator = (
                    weight_per_distance * remaining_distance[0] + speed_with_item
                ) * (
                    weight_per_distance
                    * (remaining_distance[0] - remaining_distance[item_position])
                    + speed_without_item
                )
                denominator = (
                    weight_per_distance
                    * (remaining_distance[0] - remaining_distance[item_position])
                    + speed_with_item
                ) * (
                    weight_per_distance * remaining_distance[0] + speed_without_item
                )

                # t3 usa log(numerator / denominator); se evalúa solo si es válido.
                if numerator > 0.0 and denominator > 0.0:
                    t3 = (1.0 / weight_per_distance) * math.log(
                        numerator / denominator
                    )
                    accept = item.profit > rent_per_time * t3

            if accept:
                packing[item_id] = 1
                current_weight += item_weight
                inserted_items.append(item_id)

        # En instancias muy grandes se omite eliminación para abaratar arranque.
        if inst.m_items > 100000 or inst.n_cities > 50000:
            return packing

        return self._eliminate_one_best_inserted_item(
            inst,
            tour,
            packing,
            inserted_items,
        )

    def _eliminate_one_best_inserted_item(
        self,
        inst,
        tour: list[int],
        packing: list[int],
        inserted_items: list[int],
    ) -> list[int]:
        """Evalúa eliminar un ítem insertado y aplica la mejor mejora.

        Args:
            inst: Instancia TTP de entrada.
            tour: Tour actual de ciudades.
            packing: Packing binario actual.
            inserted_items: Ítems insertados previamente.

        Returns:
            Packing actualizado tras eliminar el mejor candidato, si mejora.
        """
        if inst.distance_matrix is None:
            inst.create_distance_matrix()

        # Estado completo para evaluar eliminaciones de forma consistente.
        cache = recompute_history_full(inst, tour, list(packing))
        distance_matrix = inst.distance_matrix
        rent_per_time = inst.rent_per_time

        best_gain = cache.G_gain
        best_item_id = None

        for item_id in inserted_items:
            if not cache.packing[item_id]:
                # Puede ocurrir si el packing fue alterado antes de esta evaluación.
                continue

            item = inst.items[item_id]
            item_position = cache.pos_in_tour[item.city_id]

            # Profit tras eliminar el ítem candidato.
            candidate_profit = cache.g_profit - item.profit

            # El prefijo anterior a la ciudad del ítem no cambia.
            candidate_time = 0.0 if item_position == 0 else cache.t_acc[item_position - 1]

            # Recalcula desde la ciudad del ítem hasta cerrar el ciclo sin ese peso.
            for route_index in range(item_position, len(tour)):
                candidate_weight = cache.w_acc[route_index] - item.weight
                candidate_time += (
                    float(
                        distance_matrix[
                            tour[route_index],
                            tour[(route_index + 1) % len(tour)],
                        ]
                    )
                    / self._speed_from_weight(inst, candidate_weight)
                )

            raw_gain = candidate_profit - candidate_time * rent_per_time

            # Redondea el gain antes de comparar candidatos de eliminación.
            candidate_gain = float(math.floor(raw_gain + 0.5))

            if candidate_gain > best_gain:
                best_gain = candidate_gain
                best_item_id = item_id

        if best_item_id is not None:
            # Aplica únicamente la mejor eliminación encontrada.
            packing[best_item_id] = 0

        return packing

    def _boost_small_instance(
        self,
        inst,
        tour: list[int],
        packing: list[int],
    ) -> list[int]:
        """Mejora el packing con flips binarios en instancias pequeñas.

        Args:
            inst: Instancia TTP de entrada.
            tour: Tour actual de ciudades.
            packing: Packing binario inicial.

        Returns:
            Packing mejorado tras búsqueda local por bit-flips.
        """
        if inst.distance_matrix is None:
            inst.create_distance_matrix()

        # Caché completa para evaluar flips sin recomputar todo en cada candidato.
        cache = recompute_history_full(inst, tour, list(packing))
        distance_matrix = inst.distance_matrix
        rent_per_time = inst.rent_per_time

        improved = True

        while improved:
            improved = False
            best_gain = cache.G_gain
            best_item_id = None

            for item_id, item in enumerate(inst.items):
                current_value = cache.packing[item_id]

                # No se inserta un ítem si viola capacidad.
                if current_value == 0 and (cache.g_weight + item.weight) > inst.capacity:
                    continue

                # Flip 0->1 inserta; flip 1->0 elimina. Los deltas cambian de signo.
                delta_profit = item.profit if current_value == 0 else -item.profit
                delta_weight = item.weight if current_value == 0 else -item.weight
                candidate_profit = cache.g_profit + delta_profit

                item_position = cache.pos_in_tour[item.city_id]

                # La parte anterior a la ciudad del ítem no cambia con el flip.
                candidate_time = (
                    0.0 if item_position == 0 else cache.t_acc[item_position - 1]
                )

                # Recalcula desde la ciudad del ítem hasta el cierre con el peso nuevo.
                for route_index in range(item_position, len(tour)):
                    candidate_weight = cache.w_acc[route_index] + delta_weight
                    candidate_time += (
                        float(
                            distance_matrix[
                                tour[route_index],
                                tour[(route_index + 1) % len(tour)],
                            ]
                        )
                        / self._speed_from_weight(inst, candidate_weight)
                    )

                candidate_gain = candidate_profit - candidate_time * rent_per_time

                if candidate_gain > best_gain:
                    best_gain = candidate_gain
                    best_item_id = item_id
                    improved = True

            if improved and best_item_id is not None:
                # Aplica solo el mejor flip de la pasada y actualiza incrementalmente.
                current_value = cache.packing[best_item_id]
                incremental_bitflip(inst, cache, best_item_id, 1 - current_value)

        return cache.packing

    def _build_initial_packing(
        self,
        inst,
        tour: list[int],
        verbose: bool = False,
        log_fn: Callable[[str], None] | None = None,
    ) -> list[int]:
        """Construye el packing inicial del solver.

        Args:
            inst: Instancia TTP de entrada.
            tour: Tour TSP inicial.
            verbose: Si es True, registra la estrategia usada.
            log_fn: Función opcional de logging.

        Returns:
            Packing binario inicial.
        """
        if log_fn is None:
            log_fn = logger.info

        # Instancias pequeñas/medias usan inserción + eliminación; instancias
        # grandes usan solo t2 para abaratar inicialización.
        if inst.n_cities < INIT_ELIMINATION_CITY_CUTOFF:
            packing = self._insert_and_eliminate_packing(inst, tour)

            if verbose:
                log_fn("[INIT] packing strategy=insert_and_eliminate")
        else:
            packing, _ = self._insert_t2_packing(inst, tour)

            if verbose:
                log_fn("[INIT] packing strategy=insert_t2_only")

        # El boost por bit-flip solo se aplica a instancias pequeñas.
        if inst.n_cities < INIT_BOOST_CITY_CUTOFF:
            if verbose:
                log_fn("[INIT] boost_small_instance enabled")

            packing = self._boost_small_instance(inst, tour, packing)

        return packing

    def _build_random_restart_packing(self, inst) -> list[int]:
        """Construye un packing aleatorio para reinicios.

        Args:
            inst: Instancia TTP de entrada.

        Returns:
            Packing binario generado en orden aleatorio hasta alcanzar capacidad.
        """
        n_items = inst.m_items
        capacity = inst.capacity
        items = inst.items

        packing = [0] * n_items
        item_order = list(range(n_items))
        self.rnd.shuffle(item_order)

        current_weight = 0.0

        for item_id in item_order:
            item_weight = items[item_id].weight

            if current_weight + item_weight <= capacity:
                packing[item_id] = 1
                current_weight += item_weight
            else:
                # El reinicio agrega ítems en orden aleatorio hasta encontrar uno que no cabe.
                break

        return packing

    def build_initial(
        self,
        inst,
        verbose: bool = False,
        log_fn: Callable[[str], None] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Construye la solución inicial del solver.

        Args:
            inst: Instancia TTP de entrada.
            verbose: Si es True, registra trazas de inicialización.
            log_fn: Función opcional de logging.

        Returns:
            Tupla ``(tour, packing)`` con tour TSP inicial y packing inicial.
        """
        if log_fn is None:
            log_fn = logger.info

        if verbose:
            log_fn(f"[INIT] build_initial start n={inst.n_cities} m={inst.m_items}")

        tour = self._lk_heuristic_tour(
            inst,
            start=self.start_city,
            verbose=verbose,
            log_fn=log_fn,
        )
        packing = self._build_initial_packing(
            inst,
            tour,
            verbose=verbose,
            log_fn=log_fn,
        )

        if verbose:
            log_fn("[INIT] build_initial done")

        return tour, packing

    def build_restart(
        self,
        inst,
        verbose: bool = False,
        log_fn: Callable[[str], None] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Construye una solución para la fase de reinicio.

        Args:
            inst: Instancia TTP de entrada.
            verbose: Si es True, registra trazas del reinicio.
            log_fn: Función opcional de logging.

        Returns:
            Tupla ``(tour, packing)`` con tour nuevo y packing aleatorio.
        """
        if log_fn is None:
            log_fn = logger.info

        if verbose:
            log_fn(f"[INIT] build_restart start n={inst.n_cities} m={inst.m_items}")

        tour = self._lk_heuristic_tour(
            inst,
            start=self.start_city,
            verbose=verbose,
            log_fn=log_fn,
        )

        packing = self._build_random_restart_packing(inst)

        if verbose:
            cache = recompute_history_full(inst, tour, packing)
            log_fn(
                f"[INIT] restart ready picked={sum(packing)} "
                f"gain={cache.G_gain:.4f} time={cache.f_time:.4f} "
                f"profit={cache.g_profit:.4f}"
            )

        return tour, packing