#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/optimization/neural/inference.py

"""Inferencia neuronal tour-only y construcción de soluciones TTP.

Este módulo convierte una instancia TTP en un batch compatible con el modelo,
ejecuta el forward neuronal, decodifica un tour desde logits o probabilidades y,
opcionalmente, construye una solución TTP completa optimizando el packing sobre
el tour predicho.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Union

import torch

from src.ttp_packages.domain.solution import TTPSolution
from src.ttp_packages.infrastructure.runtime import get_default_device
from src.ttp_packages.ml_data.torch.transforms.augment import augment_batch_on_device
from src.ttp_packages.optimization.classical.packing.api import solve_pack_for_fixed_tour

from .config import MASK_MODE_DENSE
from .decoding import (
    decode_tours_greedy_from_logits,
    decode_tours_greedy_from_probs,
    repair_tour,
)


def _instance_to_batch(inst: Any, device: torch.device) -> Dict[str, Any]:
    """Convierte una instancia TTP a un batch ``B=1``.

    Args:
        inst: Instancia TTP con ciudades, ítems y parámetros globales.
        device: Dispositivo donde se crearán los tensores.

    Returns:
        Batch compatible con ``augment_batch_on_device`` y ``TTPModel.forward``.
    """
    n_cities = int(inst.n_cities)
    m_items = int(inst.m_items)

    coords_raw = torch.tensor(
        [(city.x, city.y) for city in inst.cities],
        dtype=torch.float32,
        device=device,
    ).view(1, n_cities, 2)

    if m_items > 0:
        item_city = torch.tensor(
            [item.city_id for item in inst.items],
            dtype=torch.long,
            device=device,
        ).view(1, m_items)
        item_profit = torch.tensor(
            [item.profit for item in inst.items],
            dtype=torch.float32,
            device=device,
        ).view(1, m_items)
        item_weight = torch.tensor(
            [item.weight for item in inst.items],
            dtype=torch.float32,
            device=device,
        ).view(1, m_items)
    else:
        # Mantiene shapes válidos aunque la instancia no tenga ítems.
        item_city = torch.empty((1, 0), dtype=torch.long, device=device)
        item_profit = torch.empty((1, 0), dtype=torch.float32, device=device)
        item_weight = torch.empty((1, 0), dtype=torch.float32, device=device)

    capacity = torch.tensor(
        [[float(inst.capacity)]],
        dtype=torch.float32,
        device=device,
    )
    min_speed = torch.tensor(
        [[float(inst.min_speed)]],
        dtype=torch.float32,
        device=device,
    )
    max_speed = torch.tensor(
        [[float(inst.max_speed)]],
        dtype=torch.float32,
        device=device,
    )
    rent_per_time = torch.tensor(
        [[float(inst.rent_per_time)]],
        dtype=torch.float32,
        device=device,
    )

    return {
        "meta": {
            "n_cities": n_cities,
            "m_items": m_items,
            "name": getattr(inst, "name", ""),
        },
        "inputs": {
            "coords_raw": coords_raw,
            "W": capacity,
            "item_city": item_city,
            "item_profit": item_profit,
            "item_weight": item_weight,
            "min_speed": min_speed,
            "max_speed": max_speed,
            "rent_per_time": rent_per_time,
        },
        "teacher": {},
    }


def _resolve_device(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]],
) -> torch.device:
    """Resuelve el dispositivo de inferencia.

    Args:
        model: Modelo PyTorch usado para inferencia.
        device: Dispositivo explícito opcional.

    Returns:
        Dispositivo PyTorch final.
    """
    if device is not None:
        return torch.device(device)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return get_default_device()


def _warn_if_missing_train_params(model: torch.nn.Module) -> None:
    """Emite advertencia si el modelo no trae parámetros de entrenamiento.

    Args:
        model: Modelo PyTorch usado para inferencia.
    """
    train_params = getattr(model, "train_params", None)

    if train_params is None:
        warnings.warn(
            "El modelo no contiene train_params. "
            "La inferencia usará valores por defecto para mask_mode, knn_k, "
            "allow_self, sym y compute_dist_matrix. Lo ideal es cargar el modelo "
            "desde un checkpoint que conserve los parámetros usados durante entrenamiento.",
            RuntimeWarning,
            stacklevel=2,
        )


def _get_model_train_param(
    model: torch.nn.Module,
    name: str,
    default: Any,
) -> Any:
    """Obtiene un parámetro de entrenamiento guardado en el modelo.

    Soporta tanto ``dict`` como dataclasses/objetos con atributos.

    Args:
        model: Modelo PyTorch.
        name: Nombre del parámetro.
        default: Valor por defecto si el parámetro no existe.

    Returns:
        Valor del parámetro o ``default``.
    """
    train_params = getattr(model, "train_params", None)

    if isinstance(train_params, dict):
        return train_params.get(name, default)

    if train_params is not None and hasattr(train_params, name):
        return getattr(train_params, name)

    return default


def _model_requires_dist_matrix(model: torch.nn.Module) -> bool:
    """Indica si el modelo requiere ``dist_matrix`` durante inferencia.

    Args:
        model: Modelo PyTorch.

    Returns:
        True si ``edge_feature_mode == "distance_v1"``.
    """
    edge_feature_mode = getattr(model, "edge_feature_mode", None)

    if edge_feature_mode is None:
        params = getattr(model, "params", None)

        if isinstance(params, dict):
            edge_feature_mode = params.get("edge_feature_mode", "none")
        elif params is not None:
            edge_feature_mode = getattr(params, "edge_feature_mode", "none")
        else:
            edge_feature_mode = "none"

    return str(edge_feature_mode) == "distance_v1"


def _resolve_decoder_source(decoder_source: str) -> str:
    """Normaliza y valida la fuente interna del decoder.

    La etiqueta ``"probits"`` se acepta como nombre público para los reportes,
    pero se traduce internamente a ``"probs"`` porque el decoder usa
    ``transition_probs``.

    Args:
        decoder_source: Fuente solicitada: ``"logits"``, ``"probits"`` o
            ``"probs"``.

    Returns:
        Fuente interna normalizada: ``"logits"`` o ``"probs"``.

    Raises:
        ValueError: Si la fuente no está soportada.
    """
    source = str(decoder_source).strip().lower()

    if source in {"prob", "probit", "probits"}:
        source = "probs"

    if source not in {"logits", "probs"}:
        raise ValueError(
            "decoder_source inválido. Usa 'logits', 'probits' o 'probs'. "
            f"Recibido: {decoder_source!r}"
        )

    return source


@torch.no_grad()
def predict_tour(
    inst: Any,
    *,
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    start_city: int = 0,
    decoder_source: str = "logits",
    mask_mode: Optional[str] = None,
    knn_k: Optional[int] = None,
    allow_self: Optional[bool] = None,
    sym: Optional[bool] = None,
    compute_dist_matrix: Optional[bool] = None,
) -> List[int]:
    """Predice un tour ejecutando inferencia con un modelo entrenado.

    La inferencia intenta reutilizar los parámetros guardados en
    ``model.train_params`` para reconstruir máscaras de forma coherente con el
    entrenamiento. Los parámetros explícitos tienen prioridad sobre los guardados
    en el modelo.

    Args:
        inst: Instancia TTP.
        model: Modelo neuronal PyTorch entrenado.
        device: Dispositivo de inferencia. Si es ``None``, se infiere desde el
            modelo.
        start_city: Ciudad inicial del tour decodificado.
        decoder_source: Fuente usada por el decoder. Acepta ``"logits"``,
            ``"probits"`` o ``"probs"``; ``"probits"`` y ``"probs"`` usan
            probabilidades post-Sinkhorn.
        mask_mode: Modo de máscara. Si es ``None``, usa el valor guardado en el
            modelo o ``"dense"`` como fallback.
        knn_k: Número de vecinos KNN. Si es ``None``, usa el valor guardado en el
            modelo o ``10`` como fallback.
        allow_self: Si es ``None``, usa el valor guardado en el modelo o
            ``False`` como fallback.
        sym: Si es ``None``, usa el valor guardado en el modelo o ``True`` como
            fallback.
        compute_dist_matrix: Si es ``None``, usa el valor guardado en el modelo.
            Se fuerza a ``True`` cuando ``edge_feature_mode="distance_v1"``.

    Returns:
        Tour predicho y reparado como lista de ciudades.
    """
    _warn_if_missing_train_params(model)

    decoder_source = _resolve_decoder_source(decoder_source)

    resolved_mask_mode = (
        mask_mode
        if mask_mode is not None
        else _get_model_train_param(model, "mask_mode", MASK_MODE_DENSE)
    )

    resolved_knn_k_raw = (
        knn_k
        if knn_k is not None
        else _get_model_train_param(model, "knn_k", 10)
    )
    resolved_knn_k = int(resolved_knn_k_raw)

    resolved_allow_self_raw = (
        allow_self
        if allow_self is not None
        else _get_model_train_param(model, "allow_self", False)
    )
    resolved_allow_self = bool(resolved_allow_self_raw)

    resolved_sym_raw = (
        sym
        if sym is not None
        else _get_model_train_param(model, "sym", True)
    )
    resolved_sym = bool(resolved_sym_raw)

    resolved_compute_dist_matrix_raw = (
        compute_dist_matrix
        if compute_dist_matrix is not None
        else _get_model_train_param(model, "compute_dist_matrix", False)
    )
    resolved_compute_dist_matrix = bool(resolved_compute_dist_matrix_raw)

    # distance_v1 necesita dist_matrix aunque el checkpoint/config no lo haya
    # dejado explícito.
    if _model_requires_dist_matrix(model):
        resolved_compute_dist_matrix = True

    inference_device = _resolve_device(model, device)

    model.eval()
    model.to(inference_device)

    batch = _instance_to_batch(inst, device=inference_device)

    # Reconstruye las mismas vistas derivadas que usa entrenamiento:
    # coords normalizadas, máscaras y dist_matrix opcional.
    batch = augment_batch_on_device(
        batch,
        mask_mode=resolved_mask_mode,
        knn_k=resolved_knn_k,
        allow_self=resolved_allow_self,
        sym=resolved_sym,
        compute_dist_matrix=resolved_compute_dist_matrix,
        compute_tour_adj=False,
    )

    inputs = batch["inputs"]

    transition_probs, edge_logits = model(
        coords=inputs["coords"],
        W=inputs["W"],
        node_mask=inputs["node_mask"],
        sinkhorn_mask=inputs["sinkhorn_mask"],
        item_city=inputs["item_city"],
        item_profit=inputs["item_profit"],
        item_weight=inputs["item_weight"],
        item_mask=inputs["item_mask"],
        min_speed=inputs["min_speed"],
        max_speed=inputs["max_speed"],
        rent_per_time=inputs["rent_per_time"],
        dist_matrix=inputs.get("dist_matrix", None),
    )

    # decoder_mask es la máscara de inferencia. Si falta por compatibilidad con
    # un flujo antiguo, se usa sinkhorn_mask como fallback.
    allowed_mask = inputs.get("decoder_mask", inputs.get("sinkhorn_mask", None))

    if decoder_source == "logits":
        tours = decode_tours_greedy_from_logits(
            edge_logits,
            start=start_city,
            allowed_mask=allowed_mask,
            allow_self=resolved_allow_self,
        )
    else:
        tours = decode_tours_greedy_from_probs(
            transition_probs,
            start=start_city,
            allowed_mask=allowed_mask,
            allow_self=resolved_allow_self,
        )

    return repair_tour(
        tours[0],
        n_cities=int(inst.n_cities),
        start_city=start_city,
    )


@torch.no_grad()
def solve_tour_only(
    inst: Any,
    *,
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    start_city: int = 0,
    decoder_source: str = "logits",
    mask_mode: Optional[str] = None,
    knn_k: Optional[int] = None,
    allow_self: Optional[bool] = None,
    sym: Optional[bool] = None,
    compute_dist_matrix: Optional[bool] = None,
) -> TTPSolution:
    """Construye una ``TTPSolution`` usando el tour neuronal y packing vacío.

    Args:
        inst: Instancia TTP.
        model: Modelo neuronal PyTorch entrenado.
        device: Dispositivo de inferencia.
        start_city: Ciudad inicial del tour.
        decoder_source: Fuente usada por el decoder. Acepta ``"logits"``,
            ``"probits"`` o ``"probs"``; ``"probits"`` y ``"probs"`` usan
            probabilidades post-Sinkhorn.
        mask_mode: Modo de máscara opcional.
        knn_k: Número de vecinos KNN opcional.
        allow_self: Control opcional de self-loops.
        sym: Control opcional de simetría para KNN.
        compute_dist_matrix: Control opcional de cálculo de distancias.

    Returns:
        Solución TTP con tour predicho y packing vacío.
    """
    tour = predict_tour(
        inst,
        model=model,
        device=device,
        start_city=start_city,
        decoder_source=decoder_source,
        mask_mode=mask_mode,
        knn_k=knn_k,
        allow_self=allow_self,
        sym=sym,
        compute_dist_matrix=compute_dist_matrix,
    )

    packing = [0] * int(inst.m_items)
    return TTPSolution(inst, tour, packing)


@torch.no_grad()
def solve_ttp_with_model(
    inst: Any,
    *,
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    start_city: int = 0,
    decoder_source: str = "logits",
    mask_mode: Optional[str] = None,
    knn_k: Optional[int] = None,
    allow_self: Optional[bool] = None,
    sym: Optional[bool] = None,
    compute_dist_matrix: Optional[bool] = None,
    packing_time_budget_s: float = 30.0,
    packing_n_restarts: int = 3,
    packing_seed: Optional[int] = None,
    packing_verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> TTPSolution:
    """Construye una solución TTP completa usando el modelo para el tour.

    Flujo:
        1. Predice un tour con el modelo neuronal.
        2. Repara estructuralmente el tour.
        3. Optimiza el packing sobre ese tour fijo con el solver clásico.

    Args:
        inst: Instancia TTP.
        model: Modelo neuronal PyTorch entrenado.
        device: Dispositivo de inferencia.
        start_city: Ciudad inicial del tour.
        decoder_source: Fuente usada por el decoder. Acepta ``"logits"``,
            ``"probits"`` o ``"probs"``; ``"probits"`` y ``"probs"`` usan
            probabilidades post-Sinkhorn.
        mask_mode: Modo de máscara opcional.
        knn_k: Número de vecinos KNN opcional.
        allow_self: Control opcional de self-loops.
        sym: Control opcional de simetría para KNN.
        compute_dist_matrix: Control opcional de cálculo de distancias.
        packing_time_budget_s: Presupuesto de tiempo para optimizar packing.
        packing_n_restarts: Número de reinicios del solver de packing.
        packing_seed: Semilla opcional para packing.
        packing_verbose: Si es True, muestra logs del solver de packing.
        log_fn: Función opcional de logging.

    Returns:
        Solución TTP completa con tour neuronal y packing clásico optimizado.
    """
    tour = predict_tour(
        inst,
        model=model,
        device=device,
        start_city=start_city,
        decoder_source=decoder_source,
        mask_mode=mask_mode,
        knn_k=knn_k,
        allow_self=allow_self,
        sym=sym,
        compute_dist_matrix=compute_dist_matrix,
    )

    # El solver de packing y la evaluación TTP usan la matriz de distancias de
    # la instancia. La creamos aquí si el loader todavía no la generó.
    if getattr(inst, "distance_matrix", None) is None and hasattr(
        inst,
        "create_distance_matrix",
    ):
        inst.create_distance_matrix()

    return solve_pack_for_fixed_tour(
        inst,
        tour,
        time_budget_s=packing_time_budget_s,
        n_restarts=packing_n_restarts,
        seed=packing_seed,
        verbose=packing_verbose,
        log_fn=log_fn,
    )