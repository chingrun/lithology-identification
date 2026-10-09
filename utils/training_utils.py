

import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class ModelCheckpoint:


    def __init__(
        self,
        filepath: str,
        best_primary_acc: Optional[float] = None,
        monitor: str = 'val_acc',
        mode: str = 'max'
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode

        if best_primary_acc is not None:
            self.best_value = best_primary_acc
        else:
            self.best_value = -float('inf') if mode == 'max' else float('inf')

    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        current_value: float,
        scheduler: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        **kwargs
    ) -> bool:

        improved = False

        if (self.mode == 'max' and current_value > self.best_value) or \
           (self.mode == 'min' and current_value < self.best_value):

            print(f"\nValidation {self.monitor} improved from {self.best_value:.4f} to {current_value:.4f}")
            self.best_value = current_value
            improved = True

            state = {
                'net': model.state_dict(),
                'epoch': epoch,
                'acc': self.best_value,
            }

            if scheduler is not None:
                state['scheduler'] = scheduler.state_dict()

            if optimizer is not None:
                state['optimizer'] = optimizer.state_dict()

            for key, value in kwargs.items():
                state[key] = value

            Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, self.filepath)
            print(f"Saving model weights to {self.filepath}")

        return improved


class EarlyStopping:


    def __init__(self, patience: int = 10, monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else -float('inf')

    def __call__(self, current_value: float) -> bool:

        if (self.mode == 'min' and current_value < self.best_value) or \
           (self.mode == 'max' and current_value > self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class History:


    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_primary_acc': [],
            'train_secondary_acc': [],
            'train_tertiary_acc': [],
            'val_loss': [],
            'val_primary_acc': [],
            'val_secondary_acc': [],
            'val_tertiary_acc': [],
            'lr': []
        }

    def update(self, metrics: Dict[str, Any]):

        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def keys(self):

        return self.history.keys()


class AverageMeter:


    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_all(seed: int = 42) -> None:

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CurriculumStage:


    def __init__(
        self,
        stage_id: int,
        trainable: tuple,
        lr,
        max_epochs: int,
        patience: int,
        monitor: str,
    ):
        self.stage_id = stage_id
        self.trainable = tuple(trainable)
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.monitor = monitor


def paper_curriculum_stages() -> list:

    return [
        CurriculumStage(1, ("encoder", "decoder", "primary"), 1e-3, 30, 10, "primary"),
        CurriculumStage(2, ("secondary",), 1e-3, 30, 10, "secondary"),
        CurriculumStage(3, ("tertiary",), (1e-4, 1e-3), 40, 10, "tertiary"),
        CurriculumStage(4, ("encoder", "decoder", "primary", "secondary", "tertiary"), 5e-4, 100, 10, "tertiary"),
    ]


def run_curriculum_training(
    model,
    criterion,
    train_loader,
    val_loader,
    device,
    stages: Optional[list] = None,
    train_epoch: Optional[Any] = None,
    evaluate: Optional[Any] = None,
    teacher_forcing: bool = True,
    constrained: bool = True,
    taxonomy=None,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
):
  
    import copy
    from pathlib import Path

    import torch
    import torch.optim as optim

    if stages is None:
        stages = paper_curriculum_stages()
    if train_epoch is None or evaluate is None:
        raise ValueError("train_epoch and evaluate callables are required.")

    seed_all(seed)

    history: list = []
    stage_summary: list = []
    global_epoch = 0

    for stage in stages:
        trainable = set(stage.trainable)
        model.set_trainable_modules(
            encoder="encoder" in trainable,
            decoder="decoder" in trainable,
            primary="primary" in trainable,
            secondary="secondary" in trainable,
            tertiary="tertiary" in trainable,
        )
        parameters = [p for p in model.parameters() if p.requires_grad]
        base_lr = stage.lr[1] if isinstance(stage.lr, tuple) else stage.lr
        optimizer = optim.Adam(parameters, lr=base_lr)
        scheduler = None
        if isinstance(stage.lr, tuple):
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=stage.lr[0], max_lr=stage.lr[1]
            )

        best_state = copy.deepcopy(model.state_dict())
        best_score = -float("inf")
        stale_epochs = 0
        epochs_run = 0

        for epoch in range(stage.max_epochs):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                teacher_forcing=teacher_forcing,
            )
            val_loss, metrics_p, metrics_s, metrics_t = evaluate(
                model, val_loader, criterion, device,
                constrained=constrained, taxonomy=taxonomy,
            )
            monitor_map = {
                "primary": metrics_p,
                "secondary": metrics_s,
                "tertiary": metrics_t,
            }
            score = monitor_map[stage.monitor].get("f1_macro", 0.0)
            history.append(
                {
                    "epoch": global_epoch,
                    "stage": stage.stage_id,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "monitor": stage.monitor,
                    "monitor_score": score,
                    "val_primary_acc": metrics_p.get("accuracy", 0.0),
                    "val_primary_f1": metrics_p.get("f1_macro", 0.0),
                    "val_secondary_acc": metrics_s.get("accuracy", 0.0),
                    "val_secondary_f1": metrics_s.get("f1_macro", 0.0),
                    "val_tertiary_acc": metrics_t.get("accuracy", 0.0),
                    "val_tertiary_f1": metrics_t.get("f1_macro", 0.0),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            global_epoch += 1
            epochs_run += 1

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= stage.patience:
                    break
            if scheduler is not None:
                scheduler.step()

        model.load_state_dict(best_state)
        stage_summary.append(
            {
                "stage": stage.stage_id,
                "trainable": list(stage.trainable),
                "lr": stage.lr,
                "epochs_run": epochs_run,
                "best_monitor": stage.monitor,
                "best_score": best_score,
            }
        )
        if checkpoint_dir is not None:
            output = Path(checkpoint_dir) / f"stage{stage.stage_id}_best.pth"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"stage": stage.stage_id, "net": best_state}, output)

    return {"history": history, "stages": stage_summary}
