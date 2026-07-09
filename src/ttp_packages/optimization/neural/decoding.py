#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/neural/decoding.py

"""Decodificación neuronal de tours desde logits o probabilidades.

Este módulo convierte scores de aristas ``(B, N, N)`` producidos por el modelo
en tours mediante una estrategia voraz. Los scores pueden venir de:
    - ``edge_logits``: logits pre-Sinkhorn.
    - ``transition_probs``: probabilidades suaves post-Sinkhorn.

El decoder no resuelve subtours globalmente ni hace búsqueda local; solo
construye una permutación visitando ciudades no visitadas. Por seguridad,
también se incluyen utilidades para validar y reparar tours estructuralmente.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from .config import NEG_INF


def _as_bnn(x: Tensor) -> Tensor:
    """Asegura que un tensor tenga shape ``(B, N, N)``.

    Args:
        x: Tensor de logits con shape ``(N, N)`` o ``(B, N, N)``.

    Returns:
        Tensor con shape ``(B, N, N)``.

    Raises:
        ValueError: Si ``x`` no tiene 2 o 3 dimensiones.
    """
    if x.dim() == 2:
        return x.unsqueeze(0)

    if x.dim() != 3:
        raise ValueError(
            f"edge_logits debe ser (N,N) o (B,N,N). Recibido {tuple(x.shape)}."
        )

    return x


def _expand_allowed_mask(
    allowed_mask: Tensor,
    *,
    batch_size: int,
    n_cities: int,
) -> Tensor:
    """Expande una máscara de aristas a shape ``(B, N, N)``.

    Args:
        allowed_mask: Máscara con shape ``(N, N)``, ``(1, N, N)`` o
            ``(B, N, N)``.
        batch_size: Tamaño de batch requerido.
        n_cities: Número de ciudades requerido.

    Returns:
        Máscara expandida y recortada a shape ``(B, N, N)``.

    Raises:
        ValueError: Si ``allowed_mask`` no tiene 2 o 3 dimensiones.
    """
    if allowed_mask.dim() == 2:
        mask_bnn = allowed_mask.unsqueeze(0).expand(batch_size, -1, -1)
    elif allowed_mask.dim() == 3:
        # Si viene una sola máscara, se reutiliza para todo el batch.
        mask_bnn = (
            allowed_mask
            if allowed_mask.size(0) == batch_size
            else allowed_mask[:1].expand(batch_size, -1, -1)
        )
    else:
        raise ValueError(
            "allowed_mask debe ser (N,N) o (B,N,N). "
            f"Recibido {tuple(allowed_mask.shape)}."
        )

    return mask_bnn[:, :n_cities, :n_cities]


def _neg_inf_for(tensor: Tensor) -> float:
    """Devuelve un valor negativo seguro para el dtype del tensor.

    Args:
        tensor: Tensor usado como referencia de dtype.

    Returns:
        Valor muy negativo para bloquear logits.
    """
    if tensor.is_floating_point():
        return float(torch.finfo(tensor.dtype).min)

    return float(NEG_INF)


def decode_tours_greedy_from_logits(
    edge_logits: Tensor,
    *,
    start: int = 0,
    allowed_mask: Optional[Tensor] = None,
    allow_self: bool = False,
) -> List[List[int]]:
    """Decodifica tours usando una estrategia greedy sobre logits.

    El decoder camina desde ``start`` y en cada paso elige la ciudad no visitada
    con mayor logit desde la ciudad actual. Si todas las opciones quedan
    bloqueadas por máscara, usa como fallback la primera ciudad no visitada.

    Args:
        edge_logits: Logits de aristas con shape ``(N, N)`` o ``(B, N, N)``.
        start: Ciudad inicial del tour.
        allowed_mask: Máscara opcional de aristas permitidas con shape
            ``(N, N)`` o ``(B, N, N)``.
        allow_self: Si es True, permite seleccionar self-loops. Normalmente debe
            ser False para construir tours TTP/TSP válidos.

    Returns:
        Lista de tours, uno por cada elemento del batch.

    Raises:
        ValueError: Si ``edge_logits`` o ``allowed_mask`` tienen shape inválido.
    """
    logits_bnn = _as_bnn(edge_logits)
    batch_size, n_cities, _ = logits_bnn.shape

    if n_cities <= 0:
        return [[] for _ in range(batch_size)]

    mask_bnn: Optional[Tensor] = None
    if allowed_mask is not None:
        mask_bnn = _expand_allowed_mask(
            allowed_mask,
            batch_size=batch_size,
            n_cities=n_cities,
        ).to(device=logits_bnn.device)

    tours: List[List[int]] = []

    for batch_index in range(batch_size):
        logits = logits_bnn[batch_index].clone()
        neg_inf = _neg_inf_for(logits)

        if mask_bnn is not None:
            # Bloquea aristas no permitidas antes de iniciar la caminata greedy.
            batch_mask = mask_bnn[batch_index]
            logits = logits.masked_fill(batch_mask <= 0, neg_inf)

        if not allow_self:
            # Evita transiciones i -> i, que no forman parte de un tour válido.
            city_indices = torch.arange(n_cities, device=logits.device)
            logits[city_indices, city_indices] = neg_inf

        visited = torch.zeros(n_cities, dtype=torch.bool, device=logits.device)

        current_city = int(start) if 0 <= int(start) < n_cities else 0
        visited[current_city] = True
        tour = [current_city]

        for _ in range(n_cities - 1):
            scores = logits[current_city].clone()

            # Una ciudad ya visitada no puede volver a elegirse.
            scores[visited] = neg_inf

            next_city = int(torch.argmax(scores).item())

            # Fallback defensivo: si todo quedó bloqueado, continúa con la
            # primera ciudad no visitada para completar una permutación.
            if scores[next_city].item() <= neg_inf / 2:
                remaining = (~visited).nonzero(as_tuple=False).flatten()
                next_city = int(remaining[0].item())

            tour.append(next_city)
            visited[next_city] = True
            current_city = next_city

        tours.append(tour)

    return tours


def decode_tours_greedy_from_probs(
    transition_probs: Tensor,
    *,
    start: int = 0,
    allowed_mask: Optional[Tensor] = None,
    allow_self: bool = False,
) -> List[List[int]]:
    """Decodifica tours usando probabilidades de transición post-Sinkhorn.

    Esta función reutiliza el mismo decoder greedy usado para logits, pero toma
    ``transition_probs`` como score de cada arista ``i -> j``. Esto permite
    comparar dos fuentes de decoding:

    - ``decoder_source="logits"``: usa ``edge_logits``.
    - ``decoder_source="probs"``: usa ``transition_probs``.

    Args:
        transition_probs: Probabilidades de transición con shape ``(N, N)`` o
            ``(B, N, N)``.
        start: Ciudad inicial del tour.
        allowed_mask: Máscara opcional de aristas permitidas con shape
            ``(N, N)`` o ``(B, N, N)``.
        allow_self: Si es True, permite self-loops. Normalmente debe ser False.

    Returns:
        Lista de tours, uno por cada elemento del batch.
    """
    return decode_tours_greedy_from_logits(
        transition_probs,
        start=start,
        allowed_mask=allowed_mask,
        allow_self=allow_self,
    )


def validate_tour(
    tour: List[int],
    n_cities: int,
    *,
    start_city: Optional[int] = None,
) -> bool:
    """Valida si una lista representa un tour estructuralmente correcto.

    Args:
        tour: Lista de ciudades.
        n_cities: Número total de ciudades esperadas.
        start_city: Ciudad inicial esperada. Si es ``None``, no se valida inicio.

    Returns:
        True si ``tour`` contiene exactamente una vez cada ciudad en
        ``range(n_cities)`` y, opcionalmente, comienza en ``start_city``.
    """
    n_cities = int(n_cities)

    if n_cities < 0:
        return False

    if len(tour) != n_cities:
        return False

    if set(map(int, tour)) != set(range(n_cities)):
        return False

    if start_city is not None and n_cities > 0:
        if int(tour[0]) != int(start_city):
            return False

    return True


def repair_tour(
    tour: List[int],
    n_cities: int,
    *,
    start_city: int = 0,
) -> List[int]:
    """Repara un tour para que sea una permutación válida.

    Corrige problemas estructurales simples:
        - ciudades repetidas;
        - ciudades faltantes;
        - ciudades fuera de rango;
        - largo incorrecto;
        - inicio distinto de ``start_city``.

    Esta función no optimiza el tour; solo garantiza validez estructural antes
    de construir una solución TTP.

    Args:
        tour: Tour candidato.
        n_cities: Número de ciudades esperadas.
        start_city: Ciudad inicial deseada.

    Returns:
        Tour reparado como permutación de ``range(n_cities)``.
    """
    n_cities = int(n_cities)

    if n_cities <= 0:
        return []

    start_city = int(start_city)
    if start_city < 0 or start_city >= n_cities:
        start_city = 0

    seen = set()
    cleaned: List[int] = []

    for city in tour:
        city = int(city)

        if 0 <= city < n_cities and city not in seen:
            cleaned.append(city)
            seen.add(city)

    # Agrega las ciudades que no aparecieron en el tour candidato.
    missing = [city for city in range(n_cities) if city not in seen]
    repaired = cleaned + missing
    repaired = repaired[:n_cities]

    # Fuerza que el tour comience en start_city, sin cambiar el ciclo relativo
    # más de lo necesario.
    if start_city in repaired:
        start_pos = repaired.index(start_city)
        repaired = repaired[start_pos:] + repaired[:start_pos]
    else:
        repaired = [start_city] + [city for city in repaired if city != start_city]
        repaired = repaired[:n_cities]

    return repaired

