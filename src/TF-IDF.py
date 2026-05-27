import pandas as pd
import numpy as np
import time
from joblib import dump
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, classification_report, confusion_matrix
)
from scipy.sparse import save_npz  

t0 = time.time()


# пути
DATA_PATH = "/root/NLP/data/annotation.csv"

TEST_SIZE = 0.30
RANDOM_STATE = 42

SAVE_DENSE_NPY = False

VECTORIZER_PATH = "/root/NLP/models/TF-IDF/tfidf_vectorizer_full.pkl"
MODEL_PATH = "/root/NLP/models/TF-IDF/svm_tfidf_full.pkl"

EMB_SPARSE_PATH = "/root/NLP/models/TF-IDF/tfidf_embeddings_full.npz"
EMB_DENSE_PATH = "/root/NLP/models/TF-IDF/tfidf_embeddings_full.npy"
LABELS_PATH = "/root/NLP/models/TF-IDF/labels_full.csv"

# загрузка и обработка
df = pd.read_csv(DATA_PATH)

df["text"] = (
    df["text"].astype(str)
      .str.replace('"', '', regex=False)
      .str.replace("ё", "е", regex=False)
      .str.replace("\xa0", " ")
      .str.replace("\r", " ").replace("\n", " ")
      .str.replace("«", " ").replace("»", " ")
      .str.strip()
      .str.lower()
)

df["label"] = (
    df["label"].astype(str)
      .str.replace('"', '', regex=False)
      .str.replace("ё", "е", regex=False)
      .str.replace("\xa0", " ")
      .str.replace("\r", " ").replace("\n", " ")
      .str.replace("«", " ").replace("»", " ")
      .str.strip()
      .str.lower()
)

texts = df["text"].tolist()
labels = df["label"].tolist()

print(f"Полный корпус: {len(texts)} предложений")

# разбиение 0.7 / 0.3
print("\n=== Шаг 1: train=0.7 / test=0.3, метрики на test ===")

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)
print(f"Train: {len(X_train_texts)} | Test: {len(X_test_texts)}")

# матрица эмбеддингов
vectorizer = TfidfVectorizer(
    token_pattern=None,
    tokenizer=simple_preprocess,
    min_df=2,
    ngram_range=(1, 1)
)

t_emb_07 = time.perf_counter()
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)
t_emb_07 = time.perf_counter() - t_emb_07

print("\n[Embeddings 0.7/0.3] TF-IDF построен.")
print("Матрица train:", X_train.shape, "| test:", X_test.shape)
print(f"Time (TF-IDF fit+transform) = {t_emb_07:.3f} сек")

# обучение SVM + время
clf = LinearSVC(C=1.0, max_iter=200000, loss="squared_hinge")

t_svm_07 = time.perf_counter()
clf.fit(X_train, y_train)
t_svm_07 = time.perf_counter() - t_svm_07

print(f"\n[Train 0.7] Time (SVM fit) = {t_svm_07:.3f} сек")

# предсказания на 0.3
y_pred_test = clf.predict(X_test)

# метрики на 0.3
acc = accuracy_score(y_test, y_pred_test)
mcc = matthews_corrcoef(y_test, y_pred_test)
f1_w = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
f1_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
prec_w = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
rec_w = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)

print("\n=== Метрики на TEST (0.3) ===")
print(f"Accuracy    = {acc:.4f}")
print(f"MCC         = {mcc:.4f}")
print(f"F1_weighted = {f1_w:.4f}")
print(f"F1_macro    = {f1_macro:.4f}")
print(f"Precision_w = {prec_w:.4f}")
print(f"Recall_w    = {rec_w:.4f}")

print("\n=== Classification report (TEST) ===")
print(classification_report(y_test, y_pred_test, digits=4))

print("=== Confusion matrix (TEST) ===")
print(confusion_matrix(y_test, y_pred_test))

# дообучение на 100% данных и сохранение
print("\n=== Шаг 2: обучение на 100% данных и сохранение артефактов ===")

vectorizer_full = TfidfVectorizer(
    token_pattern=None,
    tokenizer=simple_preprocess,
    min_df=2,
    ngram_range=(1, 1)
)

# полная матрица эмбеддингов
t_emb_full = time.perf_counter()
X_full = vectorizer_full.fit_transform(texts)
t_emb_full = time.perf_counter() - t_emb_full

print("\n[Embeddings FULL] TF-IDF построен.")
print("Матрица full:", X_full.shape)
print(f"Time (TF-IDF fit_transform) = {t_emb_full:.3f} сек")

# полное обучение SVM
clf_full = LinearSVC(C=1.0, max_iter=200000, loss="squared_hinge")

t_svm_full = time.perf_counter()
clf_full.fit(X_full, labels)
t_svm_full = time.perf_counter() - t_svm_full

print(f"\n[Train FULL] Time (SVM fit) = {t_svm_full:.3f} сек")

# сохранение моделей
print("\n=== Сохранение эмбеддингов и меток (FULL) ===")

pd.DataFrame({"label": labels}).to_csv(LABELS_PATH, index=False)

save_npz(EMB_SPARSE_PATH, X_full)

if SAVE_DENSE_NPY:
    X_dense = X_full.toarray()
    np.save(EMB_DENSE_PATH, X_dense)
    print(f"✓ {EMB_DENSE_PATH} сохранён (DENSE)")

dump(vectorizer_full, VECTORIZER_PATH)
dump(clf_full, MODEL_PATH)

print(f"{EMB_SPARSE_PATH} сохранён (SPARSE)")
print(f"{LABELS_PATH} сохранён")
print(f"{VECTORIZER_PATH} сохранён")
print(f"{MODEL_PATH} сохранён")

print(f"\n[Total Time] {time.time() - t0:.2f} сек")
