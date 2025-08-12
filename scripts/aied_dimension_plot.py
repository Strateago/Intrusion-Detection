import os
import argparse
import numpy as np
import pacmap
import matplotlib.pyplot as plt


class PaCMAPBinaryPlotter:
    def __init__(self, x_path: str, y_path: str, out_dir: str,
                 y_npz_key: str | None = None, verbose: bool = True):
        """
        x_path: path to .npz with X (expects key 'arr_0')
        y_path: path to labels file (.npz)
        y_npz_key: key name when Y is stored in a .npz (None -> auto)
        """
        self.x_path = x_path
        self.y_path = y_path
        self.out_dir = out_dir
        self.y_npz_key = y_npz_key
        self.verbose = verbose

        os.makedirs(self.out_dir, exist_ok=True)
        self.embedding_path = os.path.join(self.out_dir, "embedding.npy")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def load_X(self) -> np.ndarray:
        self._log(f"Loading X from: {self.x_path}")
        return np.load(self.x_path)["arr_0"]

    def load_y(self) -> np.ndarray:
        self._log(f"Loading y (NPZ) from: {self.y_path}")
        data = np.load(self.y_path)
        key = self.y_npz_key
        if key is None:
            key = "y" if "y" in data.files else data.files[0]
            self._log(f"No y_npz_key provided. Using key: '{key}'")
        y = data[key]
        if y.dtype.kind in ("U", "S", "O"):
            y = np.array([int(v) for v in y.astype(str)])
        else:
            y = y.astype(int)
        return y

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

    def run(self):
        X = self.load_X()
        y = self.load_y()

        if len(X) != len(y):
            raise ValueError(f"Size mismatch: X has {len(X)} rows, y has {len(y)} rows.")

        Z = self.get_or_fit_embedding(X)

        out_img = os.path.join(self.out_dir, "pacmap_binary_npz.png")
        self.plot_binary(Z, y, out_img)

        self._log(f"Done. Image saved to: {out_img}")


def parse_args():
    parser = argparse.ArgumentParser(description="PaCMAP projection for binary classification (X and Y in NPZ).")
    parser.add_argument("--y-npz-key", default=None,
                        help="Key inside Y .npz (optional; defaults to 'y' or first key).")
    return parser.parse_args()


if __name__ == "__main__":
    # Hardcoded paths
    X_PATH = "./feature_extracted/AIED/detection/train/X_train_AVTP_Intrusion_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_True.npz"
    Y_PATH_BINARY = "./feature_extracted/AIED/detection/train/y_train_AVTP_Intrusion_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_True.npz"
    OUT_DIR = "./aied_pacmap_embeddings"

    args = parse_args()

    plotter = PaCMAPBinaryPlotter(
        x_path=X_PATH,
        y_path=Y_PATH_BINARY,
        out_dir=OUT_DIR,
        y_npz_key=args.y_npz_key,
        verbose=True
    )
    plotter.run()
