import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional


class NPZImageSaver:
    """
    Converts X(.npz) and y(.npz/.csv) to JPG images in label folders.
    Output: out_split_dir/<label_text>/<idx>.jpg
    """
    def __init__(self, x_path: str, y_path: str, out_split_dir: str, normalize: Optional[bool] = None):
        self.x_path = x_path
        self.y_path = y_path
        self.out_split_dir = out_split_dir
        self.normalize = normalize
        os.makedirs(self.out_split_dir, exist_ok=True)

    @staticmethod
    def _load_x_npz(path: str) -> np.ndarray:
        X = np.load(path)["arr_0"]
        return X

    @staticmethod
    def _load_y_npz(path: str) -> np.ndarray:
        y = np.load(path)["arr_0"]
        return y

    @staticmethod
    def _read_csv_labels(path: str) -> np.ndarray:
        """
        Robust CSV reader:
        - With header: looks for one of ['label', 'y', 'target', 'class'].
        - No header:
            - If 1 column -> that column.
            - If >1 columns -> last column.
        Returns numpy array shape (N,), items may be int or str.
        """
        label_like = {"label", "y", "target", "class"}

        def _coerce(v: str):
            v = v.strip()
            # try numeric first
            try:
                # handle 1.0 etc.
                f = float(v)
                i = int(f)
                return i if f == i else f
            except ValueError:
                return v  # keep as string

        with open(path, "r", newline="") as f:
            peek = f.readline()
            if not peek:
                raise ValueError(f"Empty CSV: {path}")
            has_header = any(tok.isalpha() for tok in peek.replace(",", " ").split())
            f.seek(0)

            reader = csv.reader(f)
            labels: List[object] = []

            if has_header:
                header = next(reader)
                header_lower = [h.strip().lower() for h in header]
                try:
                    col_idx = next(i for i, h in enumerate(header_lower) if h in label_like)
                except StopIteration:
                    col_idx = len(header_lower) - 1  # fallback: last column

                for row in reader:
                    if not row:
                        continue
                    if col_idx >= len(row):
                        raise ValueError(f"Invalid row (not enough columns) in {path}: {row}")
                    labels.append(_coerce(row[col_idx]))
            else:
                for row in reader:
                    if not row:
                        continue
                    target = row[0] if len(row) == 1 else row[-1]
                    labels.append(_coerce(target))

        return np.asarray(labels, dtype=object)

    def _load_y(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            return self._load_y_npz(path)
        if ext == ".csv":
            return self._read_csv_labels(path)
        raise ValueError(f"Unsupported y file extension: {ext} (use .npz or .csv)")

    def _auto_normalize_flag(self, X: np.ndarray) -> bool:
        if self.normalize is not None:
            return self.normalize
        min_val, max_val = X.min(), X.max()
        print(f"Data range: min={min_val}, max={max_val}")
        if 0 <= min_val and max_val <= 1:
            print("Detected normalized data in [0, 1] → will scale to [0, 255].")
            return True
        if 0 <= min_val and max_val <= 255:
            print("Detected data already in [0, 255] → no normalization needed.")
            return False
        print("Data outside common ranges → normalization will be applied.")
        return True

    @staticmethod
    def _to_uint8_image(arr: np.ndarray, do_normalize: bool) -> np.ndarray:
        a = arr.astype(np.float32)
        if do_normalize:
            a_min, a_max = float(a.min()), float(a.max())
            if a_max > a_min:
                a = (a - a_min) / (a_max - a_min) * 255.0
            else:
                a = np.zeros_like(a)
        a = np.clip(a, 0, 255).astype(np.uint8)
        return a

    @staticmethod
    def _pil_from_array(a: np.ndarray) -> Image.Image:
        if a.ndim == 2:
            return Image.fromarray(a, mode="L")
        if a.ndim == 3 and a.shape[2] == 3:
            return Image.fromarray(a, mode="RGB")
        if a.ndim == 3 and a.shape[2] == 1:
            return Image.fromarray(a[:, :, 0], mode="L")
        # fallback
        return Image.fromarray(a[..., 0] if a.ndim == 3 else a, mode="L")

    @staticmethod
    def _sanitize_label_for_path(label_obj) -> str:
        """Turn any label (int/float/str) into a safe folder name."""
        if isinstance(label_obj, (int, np.integer)):
            s = str(int(label_obj))
        elif isinstance(label_obj, float):
            s = str(label_obj).rstrip("0").rstrip(".") if "." in str(label_obj) else str(label_obj)
        else:
            s = str(label_obj)
        # Replace path separators and trim whitespace
        s = s.strip().replace(os.sep, "_")
        if not s:
            s = "_unk_"
        return s

    def run(self):
        print(f"Loading X from: {self.x_path}")
        X = self._load_x_npz(self.x_path)

        print(f"Loading y from: {self.y_path}")
        y = self._load_y(self.y_path)

        if len(X) != len(y):
            raise ValueError(f"Size mismatch: X has {len(X)} rows, y has {len(y)} rows.")

        do_norm = self._auto_normalize_flag(X)

        # Prepare folder names (text) from labels (numeric or string)
        label_texts = np.array([self._sanitize_label_for_path(v) for v in y], dtype=object)
        # Preserve order while taking uniques
        seen = set()
        unique_label_texts = [t for t in label_texts if not (t in seen or seen.add(t))]

        # Create label subfolders
        for t in unique_label_texts:
            os.makedirs(os.path.join(self.out_split_dir, t), exist_ok=True)

        # Save images
        print(f"Saving images to: {self.out_split_dir}")
        for idx in tqdm(range(len(X)), desc=f"Saving ({os.path.basename(self.out_split_dir)})"):
            img_arr = self._to_uint8_image(X[idx], do_norm)
            img = self._pil_from_array(img_arr)
            folder = label_texts[idx]
            img.save(os.path.join(self.out_split_dir, folder, f"{idx}.jpg"))


class AEIDTwoSplitsExporter:
    """
    Orchestrates TRAIN and TEST export into:
      out_dir/
        train/<label_text>/*.jpg
        test/<label_text>/*.jpg
    """
    def __init__(self, x_train: str, y_train: str, x_test: str, y_test: str, out_dir: str, normalize: Optional[bool] = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.out_dir = out_dir
        self.normalize = normalize

    def run(self):
        train_out = os.path.join(self.out_dir, "train")
        test_out  = os.path.join(self.out_dir, "test")
        os.makedirs(self.out_dir, exist_ok=True)

        saver_train = NPZImageSaver(self.x_train, self.y_train, train_out, normalize=self.normalize)
        saver_train.run()

        saver_test = NPZImageSaver(self.x_test, self.y_test, test_out, normalize=self.normalize)
        saver_test.run()

        print(f"\nDone! Images are under:\n  {train_out}\n  {test_out}")


if __name__ == "__main__":
    # Example with TOW (y as CSV or NPZ)
    X_PATH_TRAIN = "feature_extracted/TOW/classification/train/X_train_TOW_IDS_dataset_multi_class_Wsize_44_Cols_116_Wslide_1_MC_True_sumX_False.npz"
    Y_PATH_TRAIN = "feature_extracted/TOW/classification/train/y_train_TOW_IDS_dataset_multi_class_Wsize_44_Cols_116_Wslide_1_MC_True_sumX_False.csv"

    X_PATH_TEST  = "feature_extracted/TOW/classification/test/X_test_TOW_IDS_dataset_multi_class_Wsize_44_Cols_116_Wslide_1_MC_True_sumX_False.npz"
    Y_PATH_TEST  = "feature_extracted/TOW/classification/test/y_test_TOW_IDS_dataset_multi_class_Wsize_44_Cols_116_Wslide_1_MC_True_sumX_False.csv"

    OUT_DIR = "./images/tow_images"

    exporter = AEIDTwoSplitsExporter(
        x_train=X_PATH_TRAIN,
        y_train=Y_PATH_TRAIN,
        x_test=X_PATH_TEST,
        y_test=Y_PATH_TEST,
        out_dir=OUT_DIR,
        normalize=None,  # auto
    )
    exporter.run()
