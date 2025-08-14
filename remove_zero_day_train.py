import numpy as np
import pandas as pd
import argparse
import os 

def split_zero_day(x_path, y_path, attack, output_path):

    os.makedirs(output_path, exist_ok=True)

    x = np.load(x_path)['arr_0']
    y = pd.read_csv(y_path)

    # Remover o ataque zero-day do conjunto
    idx_keep = y['Class'] != attack
    x_filtered = x[idx_keep]
    y_filtered = y[idx_keep]

    # Binarizar: 0 = Normal, 1 = Ataque
    y_bin = (y_filtered['Class'] != 'Normal').astype(int)

    # Salvar arquivos
    np.savez_compressed(os.path.join(output_path, "X_train_no_zero_day.npz"), x_filtered)
    y_bin.to_csv(os.path.join(output_path, "y_train_bin_no_zero_day.csv"), index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove um ataque do conjunto de treino e salva versão binarizada.")
    parser.add_argument("--x_path", type=str, required=True, help="Caminho para o arquivo .npz com os dados X")
    parser.add_argument("--y_path", type=str, required=True, help="Caminho para o arquivo .csv com os rótulos y")
    parser.add_argument("--attack", type=str, required=True, help="Nome do ataque a ser removido")
    parser.add_argument("--output_path", type=str, required=True, help="Diretório para salvar os arquivos gerados")

    args = parser.parse_args()
    split_zero_day(args.x_path, args.y_path, args.attack, args.output_path)
