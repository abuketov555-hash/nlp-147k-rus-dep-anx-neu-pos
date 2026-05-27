import numpy as np
import pandas as pd
import time
import os
from joblib import dump
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, matthews_corrcoef,
    precision_score, recall_score, f1_score
)

# пути и настройки
DATA_PATH = "/root/NLP/data/annotation.csv"

TEST_SIZE = 0.30
RANDOM_STATE = 42

# размерность эмбеддингов и штраф ошибки SVC
EMBEDDING_DIMS = [50, 100, 150, 200, 300]
C_GRID = [0.1, 0.3, 0.5, 1.0]

# параметры W2V
W2V_WINDOW = 6
W2V_MIN_COUNT = 2
W2V_WORKERS = 48
W2V_SG = 1

# пути сохранения моделей
SAVE_ARTIFACTS = True
W2V_FULL_PATH = "/root/NLP/models/Word2Vec/w2v_full.model"
SVM_FULL_PATH = "/root/NLP/models/Word2Vec/svm_w2v_full.pkl"
EMB_FULL_PATH = "/root/NLP/models/Word2Vec/w2v_sentence_embeddings_full.npy"
LABELS_FULL_PATH = "/root/NLP/models/Word2Vec/labels_full.csv"

# очистка текста
def clean_text_series(s: pd.Series) -> pd.Series:
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

def sentence_embedding(tokens, w2v, dim: int) -> np.ndarray:
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32)

def build_embeddings(tokens_list, w2v, dim: int) -> np.ndarray:
    # np.vstack + list comprehension быстрее и проще для среднего размера корпусов
    return np.vstack([sentence_embedding(tokens, w2v, dim) for tokens in tokens_list])

t0 = time.time()

# загрузка и очистка текстов
print("\n=== Загрузка датасета ===")
t_load = time.time()

df = pd.read_csv(DATA_PATH)
df["text"] = clean_text_series(df["text"])
df["label"] = clean_text_series(df["label"])

texts = df["text"].tolist()
labels = df["label"].tolist()

print("Документов:", len(texts))
print(f"[Time] Загрузка + очистка: {time.time() - t_load:.4f} сек")

# разбиение 0.7 / 0.3
print("\n=== Train/Test Split (для метрик) ===")
t_split = time.time()

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

print("Train:", len(X_train_texts))
print("Test :", len(X_test_texts))
print(f"[Time] Split: {time.time() - t_split:.4f} сек")

train_tokens = [simple_preprocess(t) for t in X_train_texts]
test_tokens = [simple_preprocess(t) for t in X_test_texts]

# перебор размерности и штрафа C через GridSearch
print("\n=== GRIDSEARCH: Word2Vec(train) + SVM(train) -> METRICS on test ===")

results = []  # (dim, C, acc, mcc, f1w, size_mb, w2v_time, svm_time)

for dim in EMBEDDING_DIMS:
    print(f"\n---- embedding_dim = {dim} ----")

    # W2V на 0.7
    t_w2v = time.time()
    w2v = Word2Vec(
        sentences=train_tokens,
        vector_size=dim,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=W2V_WORKERS,
        sg=W2V_SG,
        epochs=50
    )
    w2v_time = time.time() - t_w2v

    # эмбеддинги 0.7 / 0.3
    t_emb = time.time()
    X_train = build_embeddings(train_tokens, w2v, dim)
    X_test = build_embeddings(test_tokens, w2v, dim)
    emb_time = time.time() - t_emb

    for C in C_GRID:
        print(f"    Проверяем C = {C} ...")

        # SVM на 0.7
        t_svm = time.time()
        clf = LinearSVC(C=C, max_iter=200000, loss="squared_hinge")
        clf.fit(X_train, y_train)
        svm_time = time.time() - t_svm

        # предсказания и метрики на 0.3
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average="weighted")

        # оценка размера матрицы в MB
        size_mb = X_train.nbytes / 1024 / 1024

        results.append((dim, C, acc, mcc, f1w, size_mb, w2v_time, emb_time, svm_time))

        print(
            f"        acc={acc:.4f}, mcc={mcc:.4f}, f1w={f1w:.4f}, "
            f"size={size_mb:.3f} MB | w2v={w2v_time:.3f}s, emb={emb_time:.3f}s, svm={svm_time:.3f}s"
        )

# выбор лучшей конфигурации по F1_weighted
print("\n=== Выбор лучшей конфигурации по F1_weighted (оценка на test) ===")

best = max(results, key=lambda x: x[4])  # x[4] = f1w

best_dim, best_C, best_acc, best_mcc, best_f1w, best_size, best_w2v_t, best_emb_t, best_svm_t = best

print("\nЛУЧШИЕ ПАРАМЕТРЫ (по F1_weighted на test):")
print(f"embedding_dim = {best_dim}")
print(f"C            = {best_C}")
print(f"Accuracy     = {best_acc:.4f}")
print(f"MCC          = {best_mcc:.4f}")
print(f"F1_weighted  = {best_f1w:.4f}")
print(f"Train emb size ≈ {best_size:.3f} MB")

# финальная оценка метрик
print("\n=== Финальная оценка (0.7/0.3) с лучшими параметрами ===")

# Word2Vec на 0.7
t_w2v_best = time.time()
w2v_best = Word2Vec(
    sentences=train_tokens,
    vector_size=best_dim,
    window=W2V_WINDOW,
    min_count=W2V_MIN_COUNT,
    workers=W2V_WORKERS,
    sg=W2V_SG,
    epochs=50
)
print(f"[Time] Word2Vec(train) best: {time.time() - t_w2v_best:.4f} сек")

# эмбеддинги 0.7 / 0.3
t_emb_best = time.time()
X_train_best = build_embeddings(train_tokens, w2v_best, best_dim)
X_test_best = build_embeddings(test_tokens, w2v_best, best_dim)
print(f"[Time] Embeddings(train/test) best: {time.time() - t_emb_best:.4f} сек")

# SVM на train
t_svm_best = time.time()
clf_best = LinearSVC(C=best_C, max_iter=200000, loss="squared_hinge")
clf_best.fit(X_train_best, y_train)
print(f"[Time] SVM(train) best: {time.time() - t_svm_best:.4f} сек")

# метрики на 0.3
y_pred_test = clf_best.predict(X_test_best)

print("\n=== Метрики на TEST (0.3) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("MCC:", matthews_corrcoef(y_test, y_pred_test))
print("F1_weighted:", f1_score(y_test, y_pred_test, average="weighted"))
print("Precision_weighted:", precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
print("Recall_weighted:", recall_score(y_test, y_pred_test, average="weighted", zero_division=0))

print("\nClassification report:")
print(classification_report(y_test, y_pred_test, digits=4, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred_test))

# обучение лучшей модели на 100% 
print("\n=== Обучение лучшей модели на 100% данных (для сохранения) ===")

all_tokens = [simple_preprocess(t) for t in texts]

# W2V на 100%
t_w2v_full = time.time()
w2v_full = Word2Vec(
    sentences=all_tokens,
    vector_size=best_dim,
    window=W2V_WINDOW,
    min_count=W2V_MIN_COUNT,
    workers=W2V_WORKERS,
    sg=W2V_SG,
    epochs=50
)
print(f"[Time] Word2Vec(full): {time.time() - t_w2v_full:.4f} сек")

# эмбеддинги 100%
t_emb_full = time.time()
X_full = build_embeddings(all_tokens, w2v_full, best_dim)
print(f"[Time] Embeddings(full): {time.time() - t_emb_full:.4f} сек")
print("X_full:", X_full.shape)

# SVM на 100%
t_svm_full = time.time()
clf_full = LinearSVC(C=best_C, max_iter=200000, loss="squared_hinge")
clf_full.fit(X_full, labels)
print(f"[Time] SVM(full): {time.time() - t_svm_full:.4f} сек")

# сохранение модели
if SAVE_ARTIFACTS:
    print("\n=== Сохранение артефактов full-модели ===")

    for path in [W2V_FULL_PATH, SVM_FULL_PATH, EMB_FULL_PATH, LABELS_FULL_PATH]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    w2v_full.save(W2V_FULL_PATH)
    dump(clf_full, SVM_FULL_PATH)
    np.save(EMB_FULL_PATH, X_full)
    pd.DataFrame({"label": labels}).to_csv(LABELS_FULL_PATH, index=False)

    print(f"✓ {W2V_FULL_PATH} сохранён")
    print(f"✓ {SVM_FULL_PATH} сохранён")
    print(f"✓ {EMB_FULL_PATH} сохранён")
    print(f"✓ {LABELS_FULL_PATH} сохранён")

print(f"\n[Total Time] {time.time() - t0:.4f} сек")
