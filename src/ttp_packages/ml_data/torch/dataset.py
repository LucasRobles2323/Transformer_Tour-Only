#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/ml_data/torch/dataset.py

"""Dataset PyTorch para payloads tensoriales TTP.

Este módulo expone ``TTPTensorDataset``, una vista indexable sobre payloads
compactos cargados desde memoria o disco.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from torch.utils.data import Dataset

from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.infrastructure.storage.dataset_io import load_dataset
from src.ttp_packages.ml_data.config import (
    KEY_COORDS_RAW,
    KEY_OBJECTIVE,
    KEY_PICKS,
    KEY_PROFIT,
    KEY_TIME,
    KEY_TOUR_NEXT,
)


logger = setup_logger(__name__)


class TTPTensorDataset(Dataset):
    """Dataset PyTorch basado en payloads tensoriales compactos.

    Attributes:
        payload: Payload original cargado en memoria.
        inputs: Diccionario de tensores de entrada.
        teacher: Diccionario de tensores target.
        names: Nombres de instancias si están disponibles.
        N: Número de ciudades por sample.
        M: Número de ítems por sample.
        S: Número total de samples.
    """

    def __init__(self, payload: Dict[str, Any]):
        """Inicializa el dataset desde un payload en memoria.

        Args:
            payload: Diccionario con ``inputs``, ``teacher``, ``n_cities`` y
                ``m_items``.
        """
        self.payload = payload
        self.inputs = payload["inputs"]
        self.teacher = payload["teacher"]
        self.names = payload.get("names", [])

        self.N = int(payload["n_cities"])
        self.M = int(payload["m_items"])
        self.S = int(payload.get("num_samples", self.inputs[KEY_COORDS_RAW].shape[0]))

    @classmethod
    def from_file(
        cls,
        file_name: str,
        verbose: bool = True,
        map_location: str = "cpu",
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> "TTPTensorDataset":
        """Carga un payload desde disco y construye el dataset.

        Args:
            file_name: Nombre del archivo de dataset.
            verbose: Si es True, registra información de carga.
            map_location: Dispositivo destino para cargar tensores.
            log_fn: Función opcional de logging.

        Returns:
            Dataset tensorial inicializado.
        """
        if log_fn is None:
            log_fn = logger.info

        if verbose:
            log_fn(f"Cargando dataset desde el archivo: {file_name}")

        payload = load_dataset(
            file_name,
            verbose=verbose,
            map_location=map_location,
            log_fn=log_fn,
        )
        return cls(payload)

    def __len__(self) -> int:
        """Devuelve la cantidad total de instancias en el dataset."""
        return self.S

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Obtiene una instancia individual del dataset.

        Args:
            idx: Índice del sample.

        Returns:
            Sample individual con secciones ``meta``, ``inputs`` y ``teacher``.
        """
        inp = {key: self.inputs[key][idx] for key in self.inputs.keys()}
        tea = {
            KEY_TOUR_NEXT: self.teacher[KEY_TOUR_NEXT][idx],
            KEY_PICKS: self.teacher[KEY_PICKS][idx],
            # Los escalares se devuelven con shape (1,) para que collate pueda concatenarlos.
            KEY_PROFIT: self.teacher[KEY_PROFIT][idx].view(1),
            KEY_TIME: self.teacher[KEY_TIME][idx].view(1),
            KEY_OBJECTIVE: self.teacher[KEY_OBJECTIVE][idx].view(1),
        }
        meta = {
            "name": self.names[idx] if idx < len(self.names) else "",
            "n_cities": self.N,
            "m_items": self.M,
            "index": int(idx),
        }

        return {"meta": meta, "inputs": inp, "teacher": tea}