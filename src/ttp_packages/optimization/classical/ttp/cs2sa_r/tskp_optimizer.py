#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/classical/ttp/cs2sa_r/tskp_optimizer.py

"""Optimizador TSKP para tour mediante búsqueda local 2-opt.

Este módulo implementa la fase TSKP de CS2SA-R. Dado un tour y un packing fijo,
busca mejoras del tour usando movimientos 2-opt restringidos por candidatos de
Delaunay.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Optional

import numpy as np

from src.ttp_packages.infrastructure.logging import setup_logger

from .config import (
    TSKP_DEFAULT_IMPROVEMENT_THRESHOLD,
    TSKP_LARGE_INSTANCE_IMPROVEMENT_THRESHOLD,
    TSKP_LARGE_INSTANCE_MIN_CITIES,
    TSKP_MEDIUM_INSTANCE_IMPROVEMENT_THRESHOLD,
    TSKP_MEDIUM_INSTANCE_MIN_ITEMS,
)
from .delta_eval import (
    apply_2opt_inplace,
    eval_time_after_2opt,
    recompute_history_full,
)


logger = setup_logger(__name__)


class TSKPOptimizer:
    """Optimizador basado en búsqueda local 2-opt para el tour.

    Attributes:
        _candidates: Lista de candidatos por ciudad construida a partir de
            triangulación de Delaunay.
        _candidate_key: Clave usada para invalidar y reconstruir candidatos
            cuando cambia la instancia.
    """

    def __init__(self):
        """Inicializa el optimizador TSKP."""
        self._candidates: Optional[list[list[int]]] = None
        self._candidate_key = None

    def _build_delaunay_candidates(self, inst) -> list[list[int]]:
        """Construye candidatos de vecindad usando triangulación de Delaunay.

        Args:
            inst: Instancia TTP con coordenadas de ciudades.

        Returns:
            Lista donde ``out[i]`` contiene las ciudades candidatas para ciudad
            ``i``.
        """
        # Import local: scipy solo se necesita si se ejecuta TSKP.
        from scipy.spatial import Delaunay

        n_cities = int(inst.n_cities)
        coords = [(city.x, city.y) for city in inst.cities]
        candidates = [set() for _ in range(n_cities)]

        points = np.array(coords)
        triangulation = Delaunay(points)

        for simplex in triangulation.simplices:
            city_a = int(simplex[0])
            city_b = int(simplex[1])
            city_c = int(simplex[2])

            # Cada triángulo aporta vecindades locales entre sus tres vértices.
            candidates[city_a].update((city_b, city_c))
            candidates[city_b].update((city_a, city_c))
            candidates[city_c].update((city_a, city_b))

        return [sorted(list(city_candidates)) for city_candidates in candidates]

    def optimize(
        self,
        inst,
        tour: list[int],
        packing: list[int],
        max_passes: Optional[int] = None,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
        deadline: Optional[float] = None,
    ) -> tuple[list[int], list[int]]:
        """Ejecuta búsqueda local 2-opt sobre el tour actual.

        Args:
            inst: Instancia TTP.
            tour: Tour inicial a optimizar.
            packing: Solución de packing asociada al tour.
            max_passes: Máximo número de pasadas completas. Si es ``None``, no
                se impone límite por pasadas.
            verbose: Si es True, emite trazas del proceso.
            log_fn: Función de logging. Si es ``None``, usa ``logger.info``.
            deadline: Tiempo límite absoluto para interrumpir la corrida.

        Returns:
            Tupla ``(tour, packing)`` con el tour optimizado y el packing sin
            cambios.
        """
        # La clave evita reutilizar candidatos generados para otra instancia.
        candidate_key = (id(inst), inst.n_cities)

        if self._candidates is None or self._candidate_key != candidate_key:
            self._candidates = self._build_delaunay_candidates(inst)
            self._candidate_key = candidate_key

        if log_fn is None:
            log_fn = logger.info

        cache = recompute_history_full(inst, tour, packing)
        n_positions = len(cache.tour)

        threshold = TSKP_DEFAULT_IMPROVEMENT_THRESHOLD
        if inst.m_items >= TSKP_MEDIUM_INSTANCE_MIN_ITEMS:
            threshold = TSKP_MEDIUM_INSTANCE_IMPROVEMENT_THRESHOLD
        if inst.n_cities >= TSKP_LARGE_INSTANCE_MIN_CITIES:
            threshold = TSKP_LARGE_INSTANCE_IMPROVEMENT_THRESHOLD

        pass_index = 0
        candidates_db = self._candidates

        while True:
            pass_index += 1

            if deadline is not None and time.time() >= deadline:
                break

            improved = False
            edge_best = -1
            j_best = -1

            base_time = cache.f_time
            best_candidate_time = base_time
            local_moves = 0

            pos_in_tour_get = cache.pos_in_tour.get
            current_tour = cache.tour

            for i in range(1, n_positions - 1):
                left_edge = i - 1
                node1 = current_tour[i]  # city_id 0-based, coherente con el dominio.

                for node2 in candidates_db[node1]:
                    j = pos_in_tour_get(node2, -1)

                    if j <= i:
                        continue

                    candidate_time = eval_time_after_2opt(
                        inst,
                        cache,
                        left_edge,
                        j,
                    )

                    # Best-improvement: guarda el mejor 2-opt de esta pasada.
                    if (candidate_time - best_candidate_time) < threshold:
                        best_candidate_time = candidate_time
                        edge_best = left_edge
                        j_best = j
                        improved = True
                        local_moves += 1

            if improved and edge_best >= 0:
                apply_2opt_inplace(cache, edge_best, j_best)

                # Tras aceptar 2-opt, se reconstruye toda la caché física.
                cache = recompute_history_full(inst, cache.tour, cache.packing)

                if verbose:
                    log_fn(
                        f"[TSKP] nb_iter={pass_index} moves={local_moves} "
                        f"time={cache.f_time:.4f} "
                        f"delta={base_time - best_candidate_time:.4f}"
                    )

                if max_passes is not None and pass_index >= max_passes:
                    break
            else:
                if verbose:
                    log_fn(f"[TSKP] no improvement (nb_iter={pass_index})")
                break

        return cache.tour[:], cache.packing[:]