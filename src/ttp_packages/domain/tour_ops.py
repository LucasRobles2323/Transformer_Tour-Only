#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/domain/tour_ops.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

DEFAULT_START_CITY = 0
ERR_TOUR_NONE = "tour is None"
ERR_TOUR_DUPLICATED = "tour has duplicated cities"
ERR_TOUR_MISSING = "tour has missing cities"
ERR_TOUR_INVALID_IDS = "tour contains invalid city ids"
ERR_TOUR_LENGTH_MISMATCH = "len(tour)={len_t} != n_cities={n}"

def canonicalize_tour(
    tour: List[int],
    n_cities: int,
    start_city: int = DEFAULT_START_CITY,
) -> List[int]:
    """Convierte un tour a la representación canónica del dominio.

    La forma canónica es una permutación de longitud ``n_cities`` que empieza en
    ``start_city`` y no repite la ciudad inicial al final.

    Args:
        tour: Secuencia de ciudades producida por un solver o heurística.
        n_cities: Cantidad total esperada de ciudades.
        start_city: Ciudad que debe quedar al inicio del tour.

    Returns:
        Tour normalizado como permutación de ``0..n_cities-1``.

    Raises:
        ValueError: Si el tour está vacío, tiene largo inválido, no contiene
            ``start_city`` o no es una permutación válida.
    """
    if tour is None or len(tour) == 0:
        raise ValueError("Tour vacío o None.")

    normalized = list(map(int, tour))

    # Muchos solvers devuelven ciclos cerrados: [0, 2, 1, 0].
    # Internamente se trabaja con la permutación abierta: [0, 2, 1].
    if len(normalized) >= 2 and normalized[0] == normalized[-1]:
        normalized = normalized[:-1]

    if len(normalized) != n_cities:
        raise ValueError(
            f"Tour inválido: esperado len={n_cities}, "
            f"recibido len={len(normalized)}. Tour[:10]={normalized[:10]}"
        )

    if start_city not in normalized:
        raise ValueError(f"start_city={start_city} no está en el tour.")

    start_idx = normalized.index(start_city)
    if start_idx != 0:
        normalized = normalized[start_idx:] + normalized[:start_idx]

    if sorted(normalized) != list(range(n_cities)):
        raise ValueError(
            "El tour no es una permutación 0..N-1; hay ciudades repetidas, "
            "faltantes o IDs inválidos."
        )

    return normalized


def normalize_tour(
    tour: List[int],
    n_cities: int,
    start_city: int = DEFAULT_START_CITY,
) -> List[int]:
    """Normaliza un tour al formato canónico usado por el dominio.

    Args:
        tour: Secuencia de ciudades.
        n_cities: Cantidad total de ciudades de la instancia.
        start_city: Ciudad inicial deseada.

    Returns:
        Tour canonizado.
    """
    return canonicalize_tour(tour, n_cities=n_cities, start_city=start_city)


def validate_tour(
    instance: Any,
    tour: List[int],
    start_city: int = DEFAULT_START_CITY,
) -> Dict[str, Any]:
    """Valida exhaustivamente un tour TTP.

    Args:
        instance: Instancia TTP o cualquier objeto con atributo ``n_cities``.
        tour: Secuencia de ciudades a validar.
        start_city: Ciudad inicial esperada.

    Returns:
        Diccionario con la validez del tour, el tour normalizado cuando aplica,
        y listas de problemas detectados.
    """
    n = int(instance.n_cities)

    out: Dict[str, Any] = {
        "is_valid": False,
        "normalized_tour": None,
        "issues": [],
        "n_cities_expected": n,
        "n_cities_received": 0 if tour is None else int(len(tour)),
        "duplicated_cities": [],
        "missing_cities": [],
        "extra_cities": [],
        "starts_at_start_city": False,
    }

    if tour is None:
        out["issues"].append(ERR_TOUR_NONE)
        return out

    normalized = list(map(int, tour))

    # Se acepta el formato de ciclo cerrado, pero se valida internamente como
    # permutación abierta.
    if len(normalized) >= 2 and normalized[0] == normalized[-1]:
        normalized = normalized[:-1]

    out["n_cities_received"] = len(normalized)

    counts = Counter(normalized)
    duplicated = sorted(city for city, count in counts.items() if count > 1)

    expected = set(range(n))
    received = set(normalized)

    missing = sorted(expected - received)
    extra = sorted(received - expected)

    out["duplicated_cities"] = duplicated
    out["missing_cities"] = missing
    out["extra_cities"] = extra

    if len(normalized) != n:
        out["issues"].append(ERR_TOUR_LENGTH_MISMATCH.format(len_t=len(normalized), n=n))
    if duplicated:
        out["issues"].append(ERR_TOUR_DUPLICATED)
    if missing:
        out["issues"].append(ERR_TOUR_MISSING)
    if extra:
        out["issues"].append(ERR_TOUR_INVALID_IDS)

    if out["issues"]:
        return out

    try:
        normalized_tour = normalize_tour(
            normalized,
            n_cities=n,
            start_city=start_city,
        )
    except ValueError as exc:
        out["issues"].append(str(exc))
        return out

    out["normalized_tour"] = normalized_tour
    out["starts_at_start_city"] = bool(normalized_tour and normalized_tour[0] == start_city)
    out["is_valid"] = True
    return out