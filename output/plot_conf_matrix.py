import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Plota métricas de um modelo a partir de arquivos CSV.")
parser.add_argument("--path", type=str, required=True, help="Caminho base onde estão os arquivos CSV.")
args = parser.parse_args()

base_path = args.path

# Arquivos esperados
metrics_dir = os.path.join(base_path, "metrics")

### 📊 1. Matriz de Confusão
try:
    conf_mat = pd.read_csv(confusion_csv, index_col=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão - RandomForest")
    plt.ylabel("Rótulo Real")
    plt.xlabel("Rótulo Previsto")
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/confusion.png", dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"[Erro ao carregar a matriz de confusão]\n{e}")

### 📈 2. Curva ROC
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

### 🧾 3. Métricas Gerais
try:
    metrics_df = pd.read_csv(metrics_csv)
    print("\nMétricas de Avaliação:")
    print(metrics_df.to_string(index=False))

    # Mostrar como tabela
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center')
    plt.title("Métricas - RandomForest")
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"[Erro ao carregar métricas]\n{e}")
