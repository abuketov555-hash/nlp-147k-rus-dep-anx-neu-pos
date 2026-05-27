import time
import numpy as np
import pandas as pd
import joblib

from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# настройки
MODEL_PATH = "/root/NLP/sbert_large_nlu_ru"
DATA_PATH = "/root/NLP/data/annotation.csv"

SBERT_BATCH_SIZE = 8
TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_FOLDS = 3

SCORING = "f1_macro"

# пути сохранения
SAVE_FULL = True
OUT_MODEL_PATH = "/root/NLP/models/SBERT/final_sbert_linearsvc.pkl"
OUT_EMB_PATH = "/root/NLP/models/SBERT/sbert_embeddings_full.npy"
OUT_LABELS_PATH = "/root/NLP/models/SBERT/labels_full.csv"

t0 = time.time()

# очистка текста
def clean_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace('"', '', regex=False)
         .str.replace("ё", "е", regex=False)
         .str.replace("\xa0", " ")
         .str.replace("\r", " ").replace("\n", " ")
         .str.replace("«", " ").replace("»", " ")
         .str.strip()
         .str.lower()
    )

# загрузка ЛОКАЛЬНОЙ SBERT-модели
print("\n=== Загрузка SBERT модели ===")
t_model = time.time()
model = SentenceTransformer(MODEL_PATH)
print(f"[Time] SBERT load: {time.time() - t_model:.2f} сек")

# загрузка корпуса
print("\n=== Загрузка датасета ===")
t_load = time.time()

df = pd.read_csv(DATA_PATH)
df["text"] = clean_series(df["text"])
df["label"] = clean_series(df["label"])

texts = df["text"].tolist()
y = df["label"].tolist()

print("Документов:", len(texts))
print(f"[Time] load+clean: {time.time() - t_load:.2f} сек")

cnt = Counter(y)
print("Class counts:", cnt)

rare = [k for k,v in cnt.items() if v < 2]
print("Too-rare classes (<2):", rare)


for i, label in enumerate(y):
    if label in rare:
        print("idx:", i, "label:", label)  # + print(texts[i]) если переменная есть

# SBERT-эмбеддинги на всем корпусе
print("\n=== Построение SBERT эмбеддингов (100% данных) ===")
t_emb = time.time()

X = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
    batch_size=SBERT_BATCH_SIZE
)

print("X:", X.shape)
print(f"[Time] embeddings(full): {time.time() - t_emb:.2f} сек")

# разбиение 0.7 / 0.3
print("\n=== Train/Test Split (0.7/0.3) ===")
t_split = time.time()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])
print(f"[Time] split: {time.time() - t_split:.3f} сек")

# перебор штрафа ошибки через GridSearch
print("\n=== GridSearchCV: LinearSVC (scoring=f1_macro) ===")
t_grid = time.time()

pipeline = Pipeline([
    ("clf", LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=200000
    ))
])

param_grid = {
    "clf__C": [0.1, 0.3, 0.5, 1.0],
    "clf__loss": ["hinge", "squared_hinge"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=CV_FOLDS,
    n_jobs=40,
    verbose=2,
    scoring=SCORING
)

grid.fit(X_train, y_train)

print("\nЛучшие параметры:", grid.best_params_)
print("Лучший CV-score (f1_macro):", f"{grid.best_score_:.4f}")
print(f"[Time] gridsearch: {time.time() - t_grid:.2f} сек")

# расчет метрик на 0.3 корпуса
print("\n=== Оценка на TEST (0.3) ===")
t_test = time.time()

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
f1_w = f1_score(y_test, y_pred, average="weighted")
f1_macro = f1_score(y_test, y_pred, average="macro")
prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy    = {acc:.4f}")
print(f"MCC         = {mcc:.4f}")
print(f"F1_weighted = {f1_w:.4f}")
print(f"F1_macro    = {f1_macro:.4f}")
print(f"Precision_w = {prec_w:.4f}")
print(f"Recall_w    = {rec_w:.4f}")

print("\nClassification report (TEST):")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

print("Confusion matrix (TEST):")
print(cm)

print(f"[Time] test eval: {time.time() - t_test:.3f} сек")

# финальное дообучение на 100% данных
print("\n=== Финальное обучение на 100% данных + сохранение ===")
t_full = time.time()

final_model = grid.best_estimator_
final_model.fit(X, y)

if SAVE_FULL:
    joblib.dump(final_model, OUT_MODEL_PATH)
    np.save(OUT_EMB_PATH, X)
    pd.DataFrame({"label": y}).to_csv(OUT_LABELS_PATH, index=False)

    print(f"Модель сохранена: {OUT_MODEL_PATH}")
    print(f"Эмбеддинги сохранены: {OUT_EMB_PATH}")
    print(f"Метки сохранены: {OUT_LABELS_PATH}")

print(f"[Time] full fit + save: {time.time() - t_full:.2f} сек")
print(f"\n[Total Time] {time.time() - t0:.2f} сек")
