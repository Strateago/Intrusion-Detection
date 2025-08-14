import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

X_train = np.load("/mnt/hdd/lfblcp/Intrusion-Detection/zero_day/data/no_C_R/X_train_no_zero_day.npz")['arr_0']
y_train = pd.read_csv("/mnt/hdd/lfblcp/Intrusion-Detection/zero_day/data/no_C_R/y_train_bin_no_zero_day.csv")

print("X_train shape:", X_train.shape)
print("Class:", np.unique(y_train, return_counts=True))

X_train_benign = X_train[y_train['Class'] == 0]

model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train_benign)

# === CARREGAR DADOS DE TESTE ===
X_test = np.load("/mnt/hdd/lfblcp/Intrusion-Detection/zero_day/data/test/C_R/X_test_C_R.npz")['arr_0']
y_test_df = pd.read_csv("/mnt/hdd/lfblcp/Intrusion-Detection/zero_day/data/test/C_R/y_test_C_R.csv")
y_test = y_test_df["Class"]

# === PREDIÇÃO ===
preds = model.predict(X_test)  # 1 = normal, -1 = anomalia
preds_binary = np.array([0 if p == 1 else 1 for p in preds])  # 0 = benigno, 1 = ataque

start_time = time.time()
preds = model.predict(X_test)  # 1 = normal, -1 = anomalia
inference_time = (time.time() - start_time) / len(X_test)  # tempo médio por amostra
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds_binary))

print("\nClassification Report:")
print(classification_report(y_test, preds_binary, target_names=["Benign", "Attack"]))

# Scores contínuos do Isolation Forest
scores = model.decision_function(X_test)  # quanto menor, mais anômalo

# AUROC e AP
auroc = roc_auc_score(y_test, -scores)  # inverte o sinal: menor = mais anômalo
ap = average_precision_score(y_test, -scores)

print("\nAUROC:", auroc)
print("Average Precision (AP):", ap)

# Calcular métricas gerais para tabela
acc = accuracy_score(y_test, preds_binary)
f1 = f1_score(y_test, preds_binary, average="macro")
prec = precision_score(y_test, preds_binary, average="macro")
rec = recall_score(y_test, preds_binary, average="macro")

# Criar DataFrame com resultados
results_df = pd.DataFrame([{
    "fold": "entire",  # ou número do fold
    "acc": acc,
    "f1": f1,
    "prec": prec,
    "recall": rec,
    "roc_auc": auroc,
    "inference_time": inference_time
}])

print("\nTabela de resultados:")
print(results_df)
