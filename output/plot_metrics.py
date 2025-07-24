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
        elif "test_metrics" in file:
            metrics_csv.append(os.path.join(metrics_dir, file))

if roc_csv is None:
    try:
        # Criar gráfico ROC separado para cada fold
        for idx in range(len(tpr_csv)):
            tpr_df = pd.read_csv(tpr_csv[idx])
            fpr_df = pd.read_csv(fpr_csv[idx])

            # Remove coluna de índice se existir (evita problemas com Unnamed: 0)
            tpr_df = tpr_df.drop(columns=[tpr_df.columns[0]], errors='ignore')
            fpr_df = fpr_df.drop(columns=[fpr_df.columns[0]], errors='ignore')

            plt.figure(figsize=(6, 5))

            for class_name in tpr_df.columns:
                # Converte para array NumPy
                tpr = tpr_df[class_name].values
                fpr = fpr_df[class_name].values

                # Garante que a curva ROC começa em (0,0) e termina em (1,1)
                tpr = np.insert(tpr, 0, 0.0)
                fpr = np.insert(fpr, 0, 0.0)
                tpr = np.append(tpr, 1.0)
                fpr = np.append(fpr, 1.0)

                plt.plot(fpr, tpr, label=f"Classe {class_name}")

            # Linha aleatória
            plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')

            plt.xlabel("FPR (Falsos Positivos)")
            plt.ylabel("TPR (Verdadeiros Positivos)")
            plt.title(f"Curva ROC - Fold {idx + 1}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{base_path}/results/ROC_fold{idx + 1}.png", dpi=300, bbox_inches='tight')
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
    metrics_df = pd.read_csv(metrics_csv[-1])
    metrics_df = metrics_df.drop(columns=["inference_time", "Unnamed: 0"])
    # Plotar como tabela
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.6 * len(metrics_df)))
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

    plt.title("Métricas por Fold")
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"[Erro ao carregar métricas]\n{e}")
