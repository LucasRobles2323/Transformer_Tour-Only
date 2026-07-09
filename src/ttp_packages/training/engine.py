#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/training/engine.py

"""Motor principal de entrenamiento tour-only."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.ttp_packages.infrastructure.logging import setup_logger
from src.ttp_packages.infrastructure.runtime import get_default_device
from src.ttp_packages.ml_data.torch.collate import collate_same_size
from src.ttp_packages.ml_data.torch.transforms.augment import augment_batch_on_device
from src.ttp_packages.training.callbacks import (
    TrainingState,
    append_epoch_history,
    build_training_summary,
    restore_best_weights,
    should_stop_early,
    update_training_state,
)
from src.ttp_packages.training.config import TrainingParams
from src.ttp_packages.training.evaluation import eval_tour_only
from src.ttp_packages.training.losses import compute_tour_loss_and_acc
from src.ttp_packages.training.utils import DEVICE_CUDA, move_to_device, seed_everything


logger = setup_logger(__name__)


def train_tour_only(
    model: torch.nn.Module,
    dataset: Dataset,
    params: TrainingParams,
    *,
    batch_sampler: Optional[Any] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    """Entrena un modelo en modo tour-only.

    Divide el dataset en train/validation, construye DataLoaders, ejecuta el
    ciclo de entrenamiento, evalúa validación por época, aplica early stopping y
    restaura los mejores pesos observados según ``val_loss``.

    Args:
        model: Modelo PyTorch a entrenar.
        dataset: Dataset tensorial TTP.
        params: Parámetros de entrenamiento.
        batch_sampler: Sampler opcional para controlar batches de entrenamiento.
            Si se entrega, reemplaza ``batch_size`` y ``shuffle`` en train.
        log_fn: Función opcional de logging. Si es ``None``, usa el logger del
            módulo.

    Returns:
        Tupla ``(model, history, summary)``:
            - ``model``: Modelo entrenado.
            - ``history``: Historial de métricas por época.
            - ``summary``: Resumen final del entrenamiento.

    Raises:
        ValueError: Si el dataset tiene menos de 2 samples.
    """
    # Fija semillas antes de crear split y DataLoaders para reproducibilidad.
    seed_everything(params.seed)

    if log_fn is None:
        log_fn = logger.info

    # Si params.device no viene definido, se intenta respetar el device actual del modelo.
    device = params.device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_default_device(prefer_cuda=True)

    device = torch.device(device)
    model.to(device)

    # AMP se activa automáticamente solo si se entrena en CUDA, salvo override explícito.
    use_amp = params.use_amp
    if use_amp is None:
        use_amp = device.type == DEVICE_CUDA

    scaler = torch.amp.GradScaler(DEVICE_CUDA, enabled=bool(use_amp))

    # Split mínimo: se exige al menos una muestra para train y una para val.
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError(
            "El dataset debe tener al menos 2 samples para realizar split train/val."
        )

    n_val = max(1, int(round(n_total * float(params.val_frac))))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    # random_split usa generator para que el split sea reproducible.
    generator = torch.Generator().manual_seed(params.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    # pin_memory mejora transferencia CPU->GPU cuando se usa CUDA.
    pin_memory = device.type == DEVICE_CUDA

    if batch_sampler is None:
        train_loader = DataLoader(
            train_ds,
            batch_size=params.batch_size,
            shuffle=params.shuffle_train,
            num_workers=params.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_same_size,
            drop_last=False,
        )
    else:
        # Si usas batch_sampler, debe indexar correctamente el dataset entregado.
        # Ojo: aquí el DataLoader recibe train_ds, no el dataset completo.
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=params.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_same_size,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_same_size,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )

    state = TrainingState()
    num_epochs = int(params.epochs)

    for epoch in range(1, num_epochs + 1):
        model.train()

        # Acumuladores ponderados por cantidad de nodos válidos.
        running_loss = 0.0
        running_correct = 0.0
        running_count = 0.0

        for batch in train_loader:
            batch = move_to_device(batch, device)

            # La augmentación reconstruye coords normalizadas, distancias y máscaras.
            # El dataset compacto guarda lo mínimo; estas features se calculan por batch.
            batch = augment_batch_on_device(
                batch,
                mask_mode=params.mask_mode,
                knn_k=params.knn_k,
                allow_self=params.allow_self,
                sym=params.sym,
                compute_dist_matrix=params.compute_dist_matrix,
                compute_tour_adj=False,
            )

            inputs = batch["inputs"]
            teacher = batch["teacher"]

            # set_to_none=True reduce escrituras de memoria y es recomendado en PyTorch.
            optimizer.zero_grad(set_to_none=True)

            # AMP reduce memoria y puede acelerar entrenamiento en CUDA.
            with torch.autocast(device_type=DEVICE_CUDA, enabled=bool(use_amp)):
                # El modelo devuelve:
                #   transition_probs: salida suave de Sinkhorn.
                #   edge_logits: logits pre-Sinkhorn usados por la CE supervisada.
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

                tour_next = teacher["tour_next"].long()
                node_mask = inputs["node_mask"].float()

                # La CE usa loss_mask, no sinkhorn_mask.
                # sinkhorn_mask:
                #   máscara real del modelo/Sinkhorn, reproducible en inferencia.
                # loss_mask:
                #   máscara usada solo para la pérdida supervisada. Puede incluir teacher_edges
                #   para evitar que un target quede bloqueado cuando se usa KNN.
                mask_for_loss = inputs.get("loss_mask", inputs.get("sinkhorn_mask", None))

                loss, correct, count = compute_tour_loss_and_acc(
                    edge_logits=edge_logits,
                    tour_next=tour_next,
                    node_mask=node_mask,
                    sinkhorn_mask=mask_for_loss,
                    apply_mask_to_loss=params.apply_mask_to_loss,
                    transition_probs=transition_probs,
                    sinkhorn_nll_weight=params.sinkhorn_nll_weight,
                    sinkhorn_nll_mask=inputs.get("sinkhorn_mask", None),
                )

            # GradScaler maneja escalado de gradientes cuando AMP está activo.
            scaler.scale(loss).backward()

            if params.grad_clip_norm is not None and params.grad_clip_norm > 0:
                # Antes de clippear, hay que desescalar gradientes si se usa AMP.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(params.grad_clip_norm),
                )

            scaler.step(optimizer)
            scaler.update()

            # count es número de nodos válidos; por eso loss se acumula ponderada.
            count_value = float(count.detach().item())
            running_loss += float(loss.detach().item()) * count_value
            running_correct += float(correct.detach().item())
            running_count += count_value

        train_loss = running_loss / max(running_count, 1.0)
        train_acc = running_correct / max(running_count, 1.0)

        # Validación sin gradientes: actualiza métricas, no pesos.
        val_loss, val_acc = eval_tour_only(
            model,
            val_loader,
            device=device,
            mask_mode=params.mask_mode,
            knn_k=params.knn_k,
            allow_self=params.allow_self,
            sym=params.sym,
            compute_dist_matrix=params.compute_dist_matrix,
            apply_mask_to_loss=params.apply_mask_to_loss,
            sinkhorn_nll_weight=params.sinkhorn_nll_weight,
        )

        lr_now = optimizer.param_groups[0]["lr"]

        if params.verbose:
            log_fn(
                f"[epoch {epoch:03d}/{num_epochs:03d}] "
                f"lr={lr_now:.2e} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        # Guarda métricas serializables para exportarlas junto con el modelo.
        append_epoch_history(
            state,
            epoch=epoch,
            lr=lr_now,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        # Actualiza mejor checkpoint, paciencia y contador de overfitting.
        update_training_state(
            state,
            model=model,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            min_delta=params.min_delta,
            overfit_min_delta=params.overfit_min_delta,
        )

        # Detiene si no mejora val_loss o si se detecta overfitting consecutivo.
        if should_stop_early(
            state,
            patience=params.patience,
            overfit_patience=params.overfit_patience,
        ):
            if params.verbose:
                log_fn(
                    "[TRAIN] Early stopping activado | "
                    f"epoch={epoch} | "
                    f"reason={state.stop_reason} | "
                    f"best_epoch={state.best_epoch} | "
                    f"best_val_loss={state.best_val_loss:.6f} | "
                    f"val_loss={val_loss:.6f} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_acc={val_acc:.6f}"
                )

            break

    # El modelo final vuelve al checkpoint con menor val_loss observado.
    restored = restore_best_weights(model, state)

    summary = build_training_summary(
        state,
        n_total=n_total,
        n_train=n_train,
        n_val=n_val,
        restored_best_weights=restored,
    )

    if params.verbose:
        log_fn(
            f"\n[DONE] "
            f"best_epoch={summary['best_epoch']} "
            f"best_val_loss={summary['best_val_loss']:.4f} "
            f"stop_reason={summary['stop_reason']}"
        )

    return model, state.history, summary