import os
import time
import math
import copy
import random
from typing import Tuple, Dict, Optional, List

import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_counts_from_targets(targets: np.ndarray) -> Dict[int, int]:
    """Return a dict {class_id: count} from a 1D array of targets."""
    uniq, cnt = np.unique(targets, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}


def class_counts_from_indices(ds: datasets.ImageFolder, indices: List[int]) -> Dict[int, int]:
    """Return class balance for a subset of an ImageFolder by indices."""
    labels = [ds[i][1] for i in indices]
    return class_counts_from_targets(np.array(labels, dtype=np.int64))


def pretty_counts(d: Dict[int, int]) -> str:
    return ", ".join([f"{k}: {v}" for k, v in sorted(d.items(), key=lambda kv: kv[0])])


def compute_class_weights_from_indices(ds: datasets.ImageFolder,
                                       indices: List[int],
                                       device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a subset of indices.
    Weight for class c = total_samples / (num_classes * count_c)
    """
    counts = class_counts_from_indices(ds, indices)
    num_classes = len(ds.classes)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_weighted_sampler_for_indices(ds: datasets.ImageFolder,
                                       indices: List[int]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so that per-sample weight is inverse to its class frequency
    within the given indices. Sampling is with replacement.
    """
    # class counts
    counts = class_counts_from_indices(ds, indices)
    # per-index weights (inverse of class count)
    sample_weights = []
    for idx in indices:
        _, c = ds[idx]
        sample_weights.append(1.0 / float(counts[c]))
    sample_weights = torch.DoubleTensor(sample_weights)
    # number of samples to draw per epoch = len(indices)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(indices), replacement=True)


class ImageFolderSubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, root: str, indices: List[int], transform=None):
        self.ds = datasets.ImageFolder(root, transform=None)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, target = self.ds[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idxs = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)
    n_test = int(math.floor(n * test_ratio))
    n_val = int(math.floor(n * val_ratio))
    test_idx = idxs[:n_test]
    val_idx = idxs[n_test:n_test + n_val]
    train_idx = idxs[n_test + n_val:]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


# ---------------------------
# Data Module
# ---------------------------
class ImageBinaryDataModule:
    """
    - Reports class balance at load for train_dir and test_dir.
    - During K-Fold: builds loaders per fold.
    - No undersampling here. Optionally supports WeightedRandomSampler on TRAIN.
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.train_dir: Optional[str] = cfg.get("train_dir")
        self.test_dir: Optional[str]  = cfg.get("test_dir")
        self.img_size = cfg.get("img_size", (224, 224))

        self.train_tf = transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize(self.img_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.base_tf = transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize(self.img_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if self.train_dir and self.test_dir:
            self.ds_train_full = datasets.ImageFolder(self.train_dir, transform=None)
            self.ds_test_full  = datasets.ImageFolder(self.test_dir, transform=None)
            self.num_classes   = len(self.ds_train_full.classes)

            print("Train class-to-index:", self.ds_train_full.class_to_idx)
            train_all_idx = list(range(len(self.ds_train_full)))
            train_counts = class_counts_from_indices(self.ds_train_full, train_all_idx)
            print(f"Train balance (all): {pretty_counts(train_counts)}")

            print("Test class-to-index:", self.ds_test_full.class_to_idx)
            test_all_idx = list(range(len(self.ds_test_full)))
            test_counts = class_counts_from_indices(self.ds_test_full, test_all_idx)
            print(f"Test balance (all):  {pretty_counts(test_counts)}")

        else:
            data_dir = self.cfg["data_dir"]
            self.ds_single = datasets.ImageFolder(data_dir, transform=None)
            self.num_classes = len(self.ds_single.classes)
            print("Class-to-index mapping:", self.ds_single.class_to_idx)
            all_idx = list(range(len(self.ds_single)))
            counts = class_counts_from_indices(self.ds_single, all_idx)
            print(f"Dataset balance (all): {pretty_counts(counts)}")

    def build_loaders_for_fold(self,
                               train_idx: List[int],
                               val_idx: List[int],
                               train_sampler: Optional[WeightedRandomSampler] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if self.train_dir and self.test_dir:
            train_set = ImageFolderSubsetWithTransform(self.train_dir, train_idx, transform=self.train_tf)
            val_set   = ImageFolderSubsetWithTransform(self.train_dir, val_idx,   transform=self.base_tf)
            test_idx  = list(range(len(self.ds_test_full)))
            test_set  = ImageFolderSubsetWithTransform(self.test_dir, test_idx,   transform=self.base_tf)
        else:
            n = len(self.ds_single)
            all_idx = np.arange(n)
            mask = np.zeros(n, dtype=bool)
            mask[train_idx] = True
            mask[val_idx] = True
            remaining = all_idx[~mask]
            t_take = max(1, int(len(remaining) * max(0.1, self.cfg.get("test_split", 0.1))))
            test_idx = remaining[:t_take].tolist()

            train_set = ImageFolderSubsetWithTransform(self.cfg["data_dir"], train_idx, transform=self.train_tf)
            val_set   = ImageFolderSubsetWithTransform(self.cfg["data_dir"], val_idx,   transform=self.base_tf)
            test_set  = ImageFolderSubsetWithTransform(self.cfg["data_dir"], test_idx,  transform=self.base_tf)

        # Train loader: sampler OR shuffle
        if train_sampler is not None:
            train_loader = DataLoader(train_set,
                                      batch_size=self.cfg["batch_size"],
                                      shuffle=False,            # sampler and shuffle cannot both be True
                                      sampler=train_sampler,
                                      num_workers=self.cfg["num_workers"],
                                      pin_memory=True)
        else:
            train_loader = DataLoader(train_set,
                                      batch_size=self.cfg["batch_size"],
                                      shuffle=True,
                                      num_workers=self.cfg["num_workers"],
                                      pin_memory=True)

        val_loader   = DataLoader(val_set, batch_size=self.cfg["batch_size"], shuffle=False,
                                  num_workers=self.cfg["num_workers"], pin_memory=True)
        test_loader  = DataLoader(test_set, batch_size=self.cfg["batch_size"], shuffle=False,
                                  num_workers=self.cfg["num_workers"], pin_memory=True)
        return train_loader, val_loader, test_loader

    def get_train_indices_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.train_dir and self.test_dir:
            n = len(self.ds_train_full)
            y = np.array([self.ds_train_full[i][1] for i in range(n)], dtype=np.int64)
            idx = np.arange(n)
            return idx, y
        else:
            n = len(self.ds_single)
            y = np.array([self.ds_single[i][1] for i in range(n)], dtype=np.int64)
            idx = np.arange(n)
            return idx, y


# ---------------------------
# Model
# ---------------------------
class MobileNetV3SmallClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = models.mobilenet_v3_small(weights=weights)
        else:
            model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)


# ---------------------------
# Trainer
# ---------------------------
class Trainer:
    def __init__(self, model: nn.Module, cfg: Dict):
        self.cfg = cfg
        self.device = cfg["device"]
        self.model = model.to(self.device)

        os.makedirs(cfg["output_dir"], exist_ok=True)

        # class-weighted loss (if provided)
        cw = cfg.get("class_weights_tensor", None)
        self.criterion = nn.CrossEntropyLoss(weight=cw)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        )
        self.best_by = cfg["best_by"]  # "f1" | "roc_auc" | "val_loss"
        self.patience = cfg["patience"]

        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_score = -1.0
        self.epochs_no_improve = 0
        self._best_initialized_for_loss = False

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        for imgs, targets in tqdm(loader, desc="Training", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
        return running_loss / len(loader.dataset)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        logits_list, targets_list = [], []
        running_loss = 0.0
        for imgs, targets in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(imgs)

            loss = self.criterion(logits, targets)
            running_loss += loss.item() * imgs.size(0)

            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())

        val_loss = running_loss / len(loader.dataset)

        logits = torch.cat(logits_list, dim=0)
        targets = torch.cat(targets_list, dim=0).numpy()

        # For binary classification, use score of class 1
        if logits.shape[1] >= 2:
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        else:
            probs = torch.softmax(logits, dim=1)[:, 0].numpy()

        preds = (probs >= 0.5).astype(np.int32)

        metrics = {
            "val_loss": val_loss,
            "acc": accuracy_score(targets, preds),
            "precision": precision_score(targets, preds, average="macro", zero_division=0),
            "recall": recall_score(targets, preds, average="macro", zero_division=0),
            "f1": f1_score(targets, preds, average="macro", zero_division=0),
        }
        if len(np.unique(targets)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(targets, probs)
            except ValueError:
                metrics["roc_auc"] = float("nan")
        else:
            metrics["roc_auc"] = float("nan")
        return metrics

    def _is_better(self, metrics: Dict[str, float]) -> bool:
        key = self.best_by
        score = metrics.get(key, float("nan"))
        if math.isnan(score):
            return False
        if key == "val_loss":
            if self.best_score < 0 and not self._best_initialized_for_loss:
                self.best_score = float("inf")
                self._best_initialized_for_loss = True
            return score < self.best_score
        else:
            return score > self.best_score

    def fit_until_patience(self, train_loader: DataLoader, val_loader: DataLoader, save_name: str):
        epoch = 0
        while True:
            epoch += 1
            train_loss = self._train_one_epoch(train_loader)
            val_metrics = self._evaluate(val_loader)

            improved = self._is_better(val_metrics)
            if improved:
                self.best_score = val_metrics[self.best_by]
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state, os.path.join(self.cfg["output_dir"], save_name))
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            print(f"[Epoch {epoch:03d}] "
                  f"TrainLoss: {train_loss:.4f} | "
                  f"ValLoss: {val_metrics['val_loss']:.4f} | "
                  f"Val Acc: {val_metrics['acc']:.4f} | "
                  f"Prec: {val_metrics['precision']:.4f} | "
                  f"Rec: {val_metrics['recall']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                  f"Patience: {self.epochs_no_improve}/{self.patience}")

            if self.epochs_no_improve >= self.patience:
                print("Early stopping: patience exhausted.")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        return self._evaluate(test_loader)

    @torch.no_grad()
    def measure_inference_time(self, loader: DataLoader, warmup_batches: int = 3) -> float:
        self.model.eval()
        # warmup
        it = iter(loader)
        for _ in range(min(warmup_batches, len(loader))):
            try:
                imgs, _ = next(it)
            except StopIteration:
                break
            _ = self.model(imgs.to(self.device, non_blocking=True))

        # timed
        total_imgs = 0
        start = time.perf_counter()
        for imgs, _ in tqdm(loader, desc="Measuring inference time", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)
            _ = self.model(imgs)
            total_imgs += imgs.size(0)
        end = time.perf_counter()
        return (end - start) * 1000.0 / max(1, total_imgs)  # ms/image


class MetricsTableImage:
    def __init__(self, out_path: str = "metrics_table.png"):
        self.out_path = out_path

    def save_rows(self, rows: List[dict], title: str = "K-Fold Metrics"):
        cols = ["fold", "acc", "f1", "prec", "recall", "roc_auc", "inference_time"]
        df = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])

        # format floats
        for c in ["acc","f1","prec","recall","roc_auc"]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df[c] = df[c].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")

        if "inference_time" in df:
            df["inference_time"] = pd.to_numeric(df["inference_time"], errors="coerce")
            df["inference_time"] = df["inference_time"].map(lambda x: f"{x:.1e}" if pd.notna(x) else "")

        fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4*len(df)))
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=10)

        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)

        plt.savefig(self.out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


# ---------------------------
# Pipeline (K-Fold + class-weighted)
# ---------------------------
class MobileNetV3Pipeline:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        set_seed(cfg["seed"])
        os.makedirs(cfg["output_dir"], exist_ok=True)

        self.data = ImageBinaryDataModule(cfg)

    def _new_model_trainer(self, class_weights_tensor: Optional[torch.Tensor]) -> Trainer:
        model = MobileNetV3SmallClassifier(
            num_classes=self.data.num_classes,
            pretrained=self.cfg["pretrained"]
        )
        cfg2 = dict(self.cfg)
        cfg2["class_weights_tensor"] = class_weights_tensor
        return Trainer(model=model, cfg=cfg2)

    def run_kfold(self):
        idx, y = self.data.get_train_indices_and_labels()

        # Stratified K-Fold if possible
        if len(np.unique(y)) > 1:
            splitter = StratifiedKFold(n_splits=self.cfg["k_folds"], shuffle=True, random_state=self.cfg["seed"])
            splits = splitter.split(idx, y)
        else:
            splitter = KFold(n_splits=self.cfg["k_folds"], shuffle=True, random_state=self.cfg["seed"])
            splits = splitter.split(idx)

        fold_rows = []

        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            train_idx = train_idx.tolist()
            val_idx   = val_idx.tolist()

            # ---- Show class balance for this fold (train/val)
            if self.train_dir_and_test():
                train_counts = class_counts_from_indices(self.data.ds_train_full, train_idx)
                val_counts   = class_counts_from_indices(self.data.ds_train_full, val_idx)
                print(f"\n===== Fold {fold_id}/{self.cfg['k_folds']} =====")
                print(f"Train balance: {pretty_counts(train_counts)}")
                print(f"Val   balance: {pretty_counts(val_counts)}")
            else:
                print(f"\n===== Fold {fold_id}/{self.cfg['k_folds']} =====")

            # ---- Build class weights from TRAIN indices (no undersampling)
            if self.train_dir_and_test():
                class_weights_tensor = compute_class_weights_from_indices(
                    self.data.ds_train_full, train_idx, device=torch.device(self.cfg["device"])
                )
            else:
                # single-dir fallback
                class_weights_tensor = compute_class_weights_from_indices(
                    self.data.ds_single, train_idx, device=torch.device(self.cfg["device"])
                )

            # ---- Optional: WeightedRandomSampler for TRAIN (disabled by default)
            if self.cfg.get("use_weighted_sampler", False):
                if self.train_dir_and_test():
                    sampler = build_weighted_sampler_for_indices(self.data.ds_train_full, train_idx)
                else:
                    sampler = build_weighted_sampler_for_indices(self.data.ds_single, train_idx)
            else:
                sampler = None

            # ---- Build loaders (with or without sampler)
            train_loader, val_loader, test_loader = self.data.build_loaders_for_fold(
                train_idx, val_idx, train_sampler=sampler
            )

            # ---- Train / Evaluate
            trainer = self._new_model_trainer(class_weights_tensor)
            save_name = f"best_mobilenetv3_small_fold{fold_id}.pt"
            trainer.fit_until_patience(train_loader, val_loader, save_name=save_name)

            test_metrics = trainer.test(test_loader)
            infer_time_ms = trainer.measure_inference_time(test_loader)

            row = {
                "fold": fold_id,
                "acc":  test_metrics["acc"],
                "f1":   test_metrics["f1"],
                "prec": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "roc_auc": test_metrics["roc_auc"],
                "inference_time": infer_time_ms / 1000.0,  # s/img
            }
            fold_rows.append(row)

        # summary row (mean ± std)
        summary = {"fold": "mean±std"}
        for k in ["acc","f1","prec","recall","roc_auc","inference_time"]:
            vals = [r[k] for r in fold_rows if isinstance(r[k], (int,float)) and not math.isnan(r[k])]
            if len(vals) > 0:
                mu, sd = float(np.mean(vals)), float(np.std(vals))
                summary[k] = f"{mu:.6f}±{sd:.6f}" if k != "inference_time" else f"{mu:.3e}±{sd:.3e}"
            else:
                summary[k] = ""

        table = MetricsTableImage(out_path=os.path.join(self.cfg["output_dir"], "kfold_metrics_table.png"))
        table.save_rows(fold_rows + [summary])
        print(f"Saved K-Fold metrics table to {os.path.join(self.cfg['output_dir'], 'kfold_metrics_table.png')}")

    def train_dir_and_test(self) -> bool:
        return bool(self.cfg.get("train_dir")) and bool(self.cfg.get("test_dir"))


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    CONFIG = {
        # data (separate folders)
        "train_dir": "./images/tow_images/train",
        "test_dir": "./images/tow_images/test",
        "img_size": (44, 116),   # keep your aspect ratio
        "batch_size": 2048,
        "num_workers": 4,

        # training
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pretrained": False,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "patience": 5,
        "best_by": "val_loss",   # "val_loss" | "f1" | "roc_auc"

        # k-fold
        "k_folds": 5,

        # sampling (disabled by default; enable to also balance batches)
        "use_weighted_sampler": True,

        # output
        "output_dir": "./mobilenetv3_runs",
    }

    torch.backends.cudnn.benchmark = True
    pipeline = MobileNetV3Pipeline(CONFIG)
    pipeline.run_kfold()
