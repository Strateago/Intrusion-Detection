import os
import numpy as np
from PIL import Image
from tqdm import tqdm


class NPZImageSaver:
    """
    Converte X(.npz) e y(.npz) em imagens JPG organizadas por rótulo.
    Saída: out_split_dir/<label>/<idx>.jpg
    """
    def __init__(self, x_path: str, y_path: str, out_split_dir: str, normalize: bool | None = None):
        self.x_path = x_path
        self.y_path = y_path
        self.out_split_dir = out_split_dir
        self.normalize = normalize
        os.makedirs(self.out_split_dir, exist_ok=True)

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
        # arr: (H, W) ou (H, W, C)
        a = arr.astype(np.float32)
        if do_normalize:
            a_min, a_max = float(a.min()), float(a.max())
            if a_max > a_min:  # evita div por zero
                a = (a - a_min) / (a_max - a_min) * 255.0
            else:
                a = np.zeros_like(a)  # constante -> vira preto
        a = np.clip(a, 0, 255).astype(np.uint8)
        return a

    @staticmethod
    def _pil_from_array(a: np.ndarray) -> Image.Image:
        # Garante modo certo: 2D -> 'L'; 3D com 3 canais -> 'RGB'; 3D com 1 canal -> 'L'
        if a.ndim == 2:
            return Image.fromarray(a, mode="L")
        if a.ndim == 3 and a.shape[2] == 3:
            return Image.fromarray(a, mode="RGB")
        if a.ndim == 3 and a.shape[2] == 1:
            return Image.fromarray(a[:, :, 0], mode="L")
        # Qualquer outra coisa: força para 'L'
        return Image.fromarray(a[..., 0] if a.ndim == 3 else a, mode="L")

    def run(self):
        print(f"Loading X from: {self.x_path}")
        X = np.load(self.x_path)["arr_0"]

        print(f"Loading y from: {self.y_path}")
        y = np.load(self.y_path)["arr_0"]

        if len(X) != len(y):
            raise ValueError(f"Size mismatch: X has {len(X)} rows, y has {len(y)} rows.")

        do_norm = self._auto_normalize_flag(X)

        # Cria subpastas por rótulo
        labels = np.unique(y)
        for label in labels:
            os.makedirs(os.path.join(self.out_split_dir, str(label)), exist_ok=True)

        # Salva imagens
        print(f"Saving images to: {self.out_split_dir}")
        for idx in tqdm(range(len(X)), desc=f"Saving ({os.path.basename(self.out_split_dir)})"):
            img_arr = self._to_uint8_image(X[idx], do_norm)
            img = self._pil_from_array(img_arr)
            label = int(y[idx])
            img.save(os.path.join(self.out_split_dir, str(label), f"{idx}.jpg"))


class AEIDTwoSplitsExporter:
    """
    Orquestra a exportação de TRAIN e TEST em uma estrutura:
      out_dir/
        train/<label>/*.jpg
        test/<label>/*.jpg
    """
    def __init__(self, x_train: str, y_train: str, x_test: str, y_test: str, out_dir: str, normalize: bool | None = None):
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
    # ====== Hardcoded paths (seus arquivos) ======
    X_PATH_TRAIN = "feature_extracted/AIED/classification/train/X_train_AVTP_Intrusion_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_False.npz"
    Y_PATH_TRAIN = "feature_extracted/AIED/classification/train/y_train_AVTP_Intrusion_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_False.npz"

    X_PATH_TEST  = "feature_extracted/AIED/classification/test/X_test_AVTP_Intrusion_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_False.npz"
    Y_PATH_TEST  = "feature_extracted/AIED/classification/test/y_test_AVTP_Intrusion_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_False.npz"

    OUT_DIR = "./images/aeid_images"  # vai criar train/ e test/ dentro

    exporter = AEIDTwoSplitsExporter(
        x_train=X_PATH_TRAIN,
        y_train=Y_PATH_TRAIN,
        x_test=X_PATH_TEST,
        y_test=Y_PATH_TEST,
        out_dir=OUT_DIR,
        normalize=None,
    )
    exporter.run()
