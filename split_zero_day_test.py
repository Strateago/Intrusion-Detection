import os
import argparse
import numpy as np
import pandas as pd

def split_attack_vs_rest(x_path, y_path, output_dir, attack_name):
    # Carregar X e y
    X = np.load(x_path)['arr_0']
    y_df = pd.read_csv(y_path)

    if 'Class' not in y_df.columns:
        raise ValueError("Coluna 'Class' não encontrada em y.")

    # ==== Separar Normal + ataque escolhido ====
    mask_attack = (y_df['Class'] == attack_name) | (y_df['Class'] == 'Normal')
    X_attack = X[mask_attack]
    y_attack_df = y_df[mask_attack].copy()
    y_attack_df['Class'] = (y_attack_df['Class'] != 'Normal').astype(int)

    # Salvar
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, f"X_test_{attack_name}.npz"), X_attack)
    y_attack_df.to_csv(os.path.join(output_dir, f"y_test_{attack_name}.csv"), index=False)

    print(f"[OK] Salvo Normal + '{attack_name}' -> {len(y_attack_df)} amostras")

    # ==== Separar Normal + todos os outros ataques ====
    mask_rest = (y_df['Class'] != attack_name)
    X_rest = X[mask_rest]
    y_rest_df = y_df[mask_rest].copy()
    y_rest_df['Class'] = (y_rest_df['Class'] != 'Normal').astype(int)

    # Salvar
    np.savez_compressed(os.path.join(output_dir, "X_test_resto.npz"), X_rest)
    y_rest_df.to_csv(os.path.join(output_dir, "y_test_resto.csv"), index=False)

    print(f"[OK] Salvo Normal + resto dos ataques -> {len(y_rest_df)} amostras")


def main():
    parser = argparse.ArgumentParser(description="Separar teste em 'ataque escolhido' e 'resto'")
    parser.add_argument('--x_path', type=str, required=True, help='Caminho para X_test.npz')
    parser.add_argument('--y_path', type=str, required=True, help='Caminho para y_test.csv')
    parser.add_argument('--output_dir', type=str, required=True, help='Diretório de saída')
    parser.add_argument('--attack', type=str, required=True, help='Nome exato do ataque a isolar')

    args = parser.parse_args()
    split_attack_vs_rest(args.x_path, args.y_path, args.output_dir, args.attack)


if __name__ == '__main__':
    main()
