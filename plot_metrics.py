import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir_exists(file_path):
    """Ensure the directory for the given file path exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def plot_confusion_matrices(metrics_dir, output_dir):
    count = 0
    for file in sorted(os.listdir(metrics_dir)):
        if "confusion" in file and file.endswith(".csv"):
            try:
                path = os.path.join(metrics_dir, file)
                conf_mat = pd.read_csv(path, index_col=0)

                plt.figure(figsize=(6, 5))
                sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
                plt.title("Confusion Matrix - RandomForest")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                plt.tight_layout()

                save_path = os.path.join(output_dir, f"confusion_{count}.png")
                ensure_dir_exists(save_path)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                count += 1
            except Exception as e:
                print(f"[Error loading confusion matrix {file}]\n{e}")

def plot_individual_roc_curves(tpr_files, fpr_files, output_dir):
    try:
        for idx, (tpr_path, fpr_path) in enumerate(zip(tpr_files, fpr_files)):
            tpr_df = pd.read_csv(tpr_path).drop(columns=lambda c: 'Unnamed' in c, errors='ignore')
            fpr_df = pd.read_csv(fpr_path).drop(columns=lambda c: 'Unnamed' in c, errors='ignore')

            plt.figure(figsize=(6, 5))

            for class_name in tpr_df.columns:
                tpr = np.insert(tpr_df[class_name].values, 0, 0.0)
                fpr = np.insert(fpr_df[class_name].values, 0, 0.0)
                tpr = np.append(tpr, 1.0)
                fpr = np.append(fpr, 1.0)

                plt.plot(fpr, tpr, label=f"Class {class_name}")

            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel("FPR (False Positive Rate)")
            plt.ylabel("TPR (True Positive Rate)")
            plt.title(f"ROC Curve - Fold {idx + 1}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"ROC_fold{idx + 1}.png")
            ensure_dir_exists(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"[Error loading individual ROC curves]\n{e}")

def plot_roc_from_file(roc_path, output_path):
    try:
        roc_df = pd.read_csv(roc_path)
        plt.figure(figsize=(6, 5))
        plt.plot(roc_df['fpr'], roc_df['tpr'], label="RandomForest")
        plt.plot([0, 1], [0, 1], 'k--', label="Random")
        plt.xlabel("FPR (False Positive Rate)")
        plt.ylabel("TPR (True Positive Rate)")
        plt.title("ROC Curve - RandomForest")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        ensure_dir_exists(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[Error loading aggregated ROC curve]\n{e}")

def plot_metrics_table(metrics_path, output_path):
    try:
        metrics_df = pd.read_csv(metrics_path).drop(columns=lambda c: 'Unnamed' in c or 'inference_time' in c, errors='ignore')
        metrics_df = metrics_df.round(6)

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
        plt.title("Metrics per Fold")
        plt.tight_layout()

        ensure_dir_exists(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[Error loading test metrics]\n{e}")

def plot_inference_time(metrics_path, output_path):
    try:
        df = pd.read_csv(metrics_path)
        if "inference_time" in df.columns:
            plt.figure(figsize=(8, 4))
            plt.plot(df["inference_time"], marker='o', linestyle='-', color='teal')
            plt.title("Inference Time per Fold")
            plt.xlabel("Fold")
            plt.ylabel("Inference Time (s)")
            plt.grid(True)
            plt.tight_layout()

            ensure_dir_exists(output_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"[Error plotting inference time]\n{e}")

def main():
    parser = argparse.ArgumentParser(description="Plot model metrics from CSV files.")
    parser.add_argument("--path", type=str, required=True, help="Base path where the CSV files are located.")
    args = parser.parse_args()

    base_path = args.path
    metrics_dir = os.path.join(base_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    output_dir = os.path.join(base_path, "results")
    os.makedirs(output_dir, exist_ok=True)

    tpr_files = []
    fpr_files = []
    metrics_files = []
    roc_file = None

    for file in sorted(os.listdir(metrics_dir)):
        full_path = os.path.join(metrics_dir, file)
        if "tpr" in file:
            tpr_files.append(full_path)
        elif "fpr" in file:
            fpr_files.append(full_path)
        elif "roc" in file:
            roc_file = full_path
        elif "test_metrics" in file:
            metrics_files.append(full_path)

    plot_confusion_matrices(metrics_dir, output_dir)

    if roc_file:
        plot_roc_from_file(roc_file, os.path.join(output_dir, "ROC.png"))
    else:
        plot_individual_roc_curves(tpr_files, fpr_files, output_dir)

    if metrics_files:
        latest_metrics = metrics_files[-1]
        plot_metrics_table(latest_metrics, os.path.join(output_dir, "metrics.png"))
        plot_inference_time(latest_metrics, os.path.join(output_dir, "inference_time.png"))

if __name__ == "__main__":
    main()
