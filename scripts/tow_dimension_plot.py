import os
import argparse
import numpy as np
import pacmap
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class PaCMAPPlotter:
    def __init__(self, x_path: str, y_path: str, out_dir: str, verbose: bool = True):
        self.x_path = x_path
        self.y_path = y_path
        self.out_dir = out_dir
        self.verbose = verbose
        os.makedirs(self.out_dir, exist_ok=True)
        self.embedding_path = os.path.join(self.out_dir, f"embedding.npy")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def load_X(self) -> np.ndarray:
        self._log(f"Loading X from: {self.x_path}")
        return np.load(self.x_path)["arr_0"]

    def load_y(self, labels_type: str) -> np.ndarray:
        self._log(f"Loading y from: {self.y_path}")
        if labels_type == "binary":
            return np.loadtxt(self.y_path, delimiter=",", skiprows=1, usecols=1, dtype=int)
        else:
            return np.genfromtxt(self.y_path, delimiter=",", skip_header=1, usecols=1, dtype=str)

    def get_or_fit_embedding(self, X: np.ndarray) -> np.ndarray:
        if os.path.exists(self.embedding_path):
            self._log(f"Loading cached embedding: {self.embedding_path}")
            return np.load(self.embedding_path)

        self._log("No cached embedding found. Training PaCMAP...")
        reducer = pacmap.PaCMAP(n_components=2, random_state=42, verbose=True)
        Z = reducer.fit_transform(X)
        np.save(self.embedding_path, Z)
        self._log(f"Embedding saved to: {self.embedding_path}")
        return Z

    def plot_binary(self, Z: np.ndarray, y: np.ndarray, out_img: str):
        self._log(f"Saving binary plot to: {out_img}")
        plt.figure(figsize=(8, 6))
        plt.scatter(Z[y == 0, 0], Z[y == 0, 1], s=1, alpha=0.5, label="Class 0")
        plt.scatter(Z[y == 1, 0], Z[y == 1, 1], s=1, alpha=0.5, label="Class 1")
        plt.title("PaCMAP projection (binary classes)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(markerscale=6, frameon=False)
        plt.savefig(out_img, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_multiclass(self, Z: np.ndarray, y: np.ndarray, out_img: str):
        self._log(f"Saving multiclass plot to: {out_img}")
        classes = np.unique(y)
        cmap = get_cmap("tab20", len(classes))

        plt.figure(figsize=(9, 7))
        for idx, cls in enumerate(classes):
            plt.scatter(Z[y == cls, 0], Z[y == cls, 1], s=1, alpha=0.6, label=str(cls), color=cmap(idx))

        plt.title("PaCMAP projection (multiclass)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(markerscale=6, frameon=False, ncol=2)
        plt.savefig(out_img, dpi=300, bbox_inches="tight")
        plt.close()

    def run(self, labels_type: str):
        X = self.load_X()
        y = self.load_y(labels_type)

        if len(X) != len(y):
            raise ValueError(f"Size mismatch: X has {len(X)} rows, y has {len(y)} rows.")

        Z = self.get_or_fit_embedding(X)

        base = os.path.splitext(os.path.basename(self.x_path))[0]
        out_img = os.path.join(self.out_dir, f"pacmap_{labels_type}.png")

        if labels_type == "binary":
            self.plot_binary(Z, y, out_img)
        else:
            self.plot_multiclass(Z, y, out_img)

        self._log(f"Done. Image saved to: {out_img}")


def parse_args():
    parser = argparse.ArgumentParser(description="PaCMAP projection with binary or multiclass labels")
    parser.add_argument("--labels-type", choices=["binary", "multiclass"], required=True,
                        help="How to read labels CSV and color the plot")
    return parser.parse_args()


if __name__ == "__main__":
    # Hardcoded paths
    X_PATH = "feature_extracted/TOW/detection/train/X_train_TOW_IDS_dataset_one_class_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_True.npz"
    Y_PATH_BINARY = "feature_extracted/TOW/detection/train/y_train_TOW_IDS_dataset_one_class_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_True.csv"
    Y_PATH_MULTICLASS = "feature_extracted/TOW/classification/train/y_train_TOW_IDS_dataset_multi_class_Wsize_44_Cols_116_Wslide_1_MC_True_sumX_False.csv"
    OUT_DIR = "./tow_pacmap_embeddings"

    type = 'multi'
    y_path = Y_PATH_BINARY if type == "binary" else Y_PATH_MULTICLASS

    plotter = PaCMAPPlotter(
        x_path=X_PATH,
        y_path=y_path,
        out_dir=OUT_DIR,
        verbose=True
    )
    plotter.run(labels_type=type)
