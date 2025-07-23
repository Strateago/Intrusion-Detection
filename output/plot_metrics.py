import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Plota métricas de um modelo a partir de arquivos CSV.")
parser.add_argument("--path", type=str, required=True, help="Caminho base onde estão os arquivos CSV.")
args = parser.parse_args()

base_path = args.path

conf = 0
tpr_csv = []
fpr_csv = []
metrics_csv = []
roc_csv = None

# Arquivos
metrics_dir = os.path.join(base_path, "metrics")

for file in sorted(os.listdir(metrics_dir)):
    if file.endswith(".csv"):
        if "confusion" in file:
            confusion_csv = os.path.join(metrics_dir, file)
            try:
                conf_mat = pd.read_csv(confusion_csv, index_col=0)
                plt.figure(figsize=(6, 5))
                sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
                plt.title("Matriz de Confusão - RandomForest")
                plt.ylabel("Rótulo Real")
                plt.xlabel("Rótulo Previsto")
                plt.tight_layout()
                plt.savefig(f"{base_path}/results/confusion_{conf}.png", dpi=300, bbox_inches='tight')
                plt.close()
                conf += 1
            except Exception as e:
                print(f"[Erro ao carregar a matriz de confusão]\n{e}")
        elif "tpr" in file:
            tpr_csv.append(os.path.join(metrics_dir, file))
        elif "fpr" in file:
            fpr_csv.append(os.path.join(metrics_dir, file))
        elif "roc" in file:
            roc_csv = os.path.join(metrics_dir, file)
        elif "metrics" in file:
            metrics_csv.append(os.path.join(metrics_dir, file))

if roc_csv is None:
    try:
        # Criar figura fora do loop para agrupar todas as curvas ROC
        plt.figure(figsize=(6, 5))

        # Lista ou mapa de cores para variar
        colors = plt.cm.viridis(np.linspace(0, 1, len(tpr_csv)))
        for idx, (tpr_file, fpr_file) in enumerate(zip(tpr_csv, fpr_csv)):
            tpr_df = pd.read_csv(tpr_file)
            fpr_df = pd.read_csv(fpr_file)
            label = f"Curva {idx + 1}"
            plt.plot(fpr_df['fpr'], tpr_df['tpr'], label=label, color=colors[idx])
        
        # Linha de referência aleatória
        plt.plot([0, 1], [0, 1], 'k--', label="Aleatório")
        plt.xlabel("FPR (Falsos Positivos)")
        plt.ylabel("TPR (Verdadeiros Positivos)")
        plt.title("Curvas ROC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{base_path}/results/ROC.png", dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"[Erro ao carregar curva ROC]\n{e}")

else:
    try:
        roc_df = pd.read_csv(roc_csv)
        plt.figure(figsize=(6, 5))
        plt.plot(roc_df['fpr'], roc_df['tpr'], label="RandomForest")
        plt.plot([0, 1], [0, 1], 'k--', label="Aleatório")
        plt.xlabel("FPR (Falsos Positivos)")
        plt.ylabel("TPR (Verdadeiros Positivos)")
        plt.title("Curva ROC - RandomForest")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{base_path}/results/ROC.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[Erro ao carregar curva ROC]\n{e}")


try:
    metrics_list = []

    for idx, file in enumerate(metrics_csv):
        df = pd.read_csv(file)
        
        # Se for uma única linha (ex: média de um fold), força a virar DataFrame com uma linha
        if df.shape[0] == 1:
            df.insert(0, 'Fold', f"Fold {idx}")
        else:
            df['Fold'] = f"Fold {idx}"
            df = df[['Fold'] + [col for col in df.columns if col != 'Fold']]  # Garantir ordem da coluna

        metrics_list.append(df)

    # Juntar todos em um único DataFrame
    metrics_df = pd.concat(metrics_list, ignore_index=True)

    # Plotar como tabela
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * len(metrics_df)))
    ax.axis('off')

    table = ax.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    plt.title("Métricas por Fold - RandomForest")
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"[Erro ao carregar métricas]\n{e}")
