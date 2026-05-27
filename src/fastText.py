import numpy as np
import pandas as pd
import time
from joblib import dump
from gensim.models import FastText
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, matthews_corrcoef,
    precision_score, recall_score, f1_score
)

# настройки
# ======================================================
DATA_PATH = "/root/NLP/data/annotation.csv"

TEST_SIZE = 0.30
RANDOM_STATE = 42

# размерность эмбеддингов и штраф ошибки SVC
EMBEDDING_DIMS = [50, 100, 150, 200, 300]
C_GRID = [0.1, 0.3, 0.5, 1.0]

# параметры FastText
WINDOW = 6
MIN_COUNT = 2
WORKERS = 48
SG = 1
MIN_N = 3
MAX_N = 6
EPOCHS = 50  

# пути сохранения моделей
SAVE_ARTIFACTS = True
FT_FULL_PATH = "/root/NLP/models/FastText/fasttext_full.model"
SVM_FULL_PATH = "/root/NLP/models/FastText/svm_fasttext_full.pkl"
X_FULL_PATH = "/root/NLP/models/FastText/fasttext_sentence_embeddings_full.npy"
LABELS_FULL_PATH = "/root/NLP/models/FastText/labels_full.csv"

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

def sentence_embedding(tokens, ft_model, dim: int) -> np.ndarray:
    vecs = [ft_model.wv[t] for t in tokens if t in ft_model.wv]
    return np.mean(vecs, axis=0).astype(np.float32) if vecs else np.zeros(dim, dtype=np.float32)

def build_sentence_matrix(tokens_list, ft_model, dim: int) -> np.ndarray:
    return np.vstack([sentence_embedding(toks, ft_model, dim) for toks in tokens_list])

def train_fasttext_gensim(tokens_list, dim: int, epochs: int) -> FastText:
    """
    Гарантированно обучает FastText ровно epochs эпох:
    build_vocab -> train(epochs=...)
    """
    model = FastText(
        vector_size=dim,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG,
        min_n=MIN_N,
        max_n=MAX_N
    )
    model.build_vocab(corpus_iterable=tokens_list)
    model.train(corpus_iterable=tokens_list, total_examples=len(tokens_list), epochs=epochs)
    return model

# загрузка корпуса
print("\n=== Загрузка датасета ===")
t_load = time.time()

df = pd.read_csv(DATA_PATH)
df["text"] = clean_series(df["text"])
df["label"] = clean_series(df["label"])

texts = df["text"].tolist()
labels = df["label"].tolist()

print("Документов:", len(texts))
print(f"[Time] load+clean: {time.time() - t_load:.4f} сек")

# разбиение 0.7 / 0.3
print("\n=== Train/Test Split (0.7/0.3) ===")
t_split = time.time()

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
)

train_tokens = [simple_preprocess(t) for t in X_train_texts]
test_tokens = [simple_preprocess(t) for t in X_test_texts]

print("Train:", len(train_tokens))
print("Test :", len(test_tokens))
print(f"[Time] split+tokenize: {time.time() - t_split:.4f} сек")

# перебор размерности и штрафа C через GridSearch
print("\n=== GRIDSEARCH: FastText + SVM (criterion=F1_weighted on TEST) ===")

results = []  # (dim, C, acc, mcc, f1w, size_mb, ft_time, emb_time, svm_time)

for dim in EMBEDDING_DIMS:
    print(f"\n---- embedding_dim = {dim} ----")

    # FastText на 0.7
    t_ft = time.time()
    ft = train_fasttext_gensim(train_tokens, dim=dim, epochs=EPOCHS)
    ft_time = time.time() - t_ft
    print(f"[Time] FastText(train, epochs={EPOCHS}): {ft_time:.4f} сек")

    # эмбеддинги 0.7 / 0.3
    t_emb = time.time()
    X_train = build_sentence_matrix(train_tokens, ft, dim)
    X_test = build_sentence_matrix(test_tokens, ft, dim)
    emb_time = time.time() - t_emb
    print(f"[Time] Embeddings(train/test): {emb_time:.4f} сек")

    for C in C_GRID:
        print(f"    Проверяем C = {C} ...")
        t_svm = time.time()

        clf = LinearSVC(C=C, max_iter=200000, loss="squared_hinge")
        clf.fit(X_train, y_train)
        svm_time = time.time() - t_svm

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average="weighted")

        size_mb = X_train.nbytes / 1024 / 1024

        results.append((dim, C, acc, mcc, f1w, size_mb, ft_time, emb_time, svm_time))

        print(
            f"        acc={acc:.4f}, mcc={mcc:.4f}, f1w={f1w:.4f}, size={size_mb:.3f} MB | "
            f"svm_fit={svm_time:.4f} сек"
        )

# выбор лучшей конфигурации по F1_weighted
print("\n=== Выбор лучшей модели по F1_weighted (на TEST) ===")
best = max(results, key=lambda x: x[4])  # x[4]=f1w
best_dim, best_C, best_acc, best_mcc, best_f1w, best_size, _, _, _ = best

print("\nЛУЧШИЕ ПАРАМЕТРЫ:")
print(f"embedding_dim = {best_dim}")
print(f"C = {best_C}")
print(f"F1_weighted (test) = {best_f1w:.4f}")
print(f"Train embeddings size ≈ {best_size:.3f} MB")

# финальная оценка метрик
print("\n=== Финальная оценка (0.7/0.3) с лучшими параметрами ===")

t_ft_best = time.time()
ft_best = train_fasttext_gensim(train_tokens, dim=best_dim, epochs=EPOCHS)
print(f"[Time] FastText(train, best, epochs={EPOCHS}): {time.time() - t_ft_best:.4f} сек")

t_emb_best = time.time()
X_train_best = build_sentence_matrix(train_tokens, ft_best, best_dim)
X_test_best = build_sentence_matrix(test_tokens, ft_best, best_dim)
print(f"[Time] Embeddings(best): {time.time() - t_emb_best:.4f} сек")

t_svm_best = time.time()
clf_best = LinearSVC(C=best_C, max_iter=200000, loss="squared_hinge")
clf_best.fit(X_train_best, y_train)
print(f"[Time] SVM(train, best): {time.time() - t_svm_best:.4f} сек")

y_pred_test = clf_best.predict(X_test_best)

print("\n=== Метрики на TEST (0.3) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("MCC:", matthews_corrcoef(y_test, y_pred_test))
print("F1_weighted:", f1_score(y_test, y_pred_test, average="weighted"))
print("F1_macro:", f1_score(y_test, y_pred_test, average="macro"))
print("Precision_weighted:", precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
print("Recall_weighted:", recall_score(y_test, y_pred_test, average="weighted", zero_division=0))

print("\nClassification report:")
print(classification_report(y_test, y_pred_test, digits=4, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred_test))

# дообучение на 100% данных
print("\n=== Финальное обучение лучшей модели на 100% данных + сохранение ===")

# токены на 100%
t_tok_full = time.time()
all_tokens = [simple_preprocess(t) for t in texts]
print(f"[Time] Tokenization(full): {time.time() - t_tok_full:.4f} сек")

# FastText на 100%
t_ft_full = time.time()
ft_full = train_fasttext_gensim(all_tokens, dim=best_dim, epochs=EPOCHS)
print(f"[Time] FastText(full, epochs={EPOCHS}): {time.time() - t_ft_full:.4f} сек")

# entence-эмбеддинги на 100%
t_emb_full = time.time()
X_full = build_sentence_matrix(all_tokens, ft_full, best_dim)
print("X_full:", X_full.shape)
print(f"[Time] Embeddings(full): {time.time() - t_emb_full:.4f} сек")

# SVM на 100%
t_svm_full = time.time()
clf_full = LinearSVC(C=best_C, max_iter=200000, loss="squared_hinge")
clf_full.fit(X_full, labels)
print(f"[Time] SVM(full): {time.time() - t_svm_full:.4f} сек")

# сохранение
if SAVE_ARTIFACTS:
    ft_full.save(FT_FULL_PATH)
    dump(clf_full, SVM_FULL_PATH)
    np.save(X_FULL_PATH, X_full)
    pd.DataFrame({"label": labels}).to_csv(LABELS_FULL_PATH, index=False)

    print(f"{FT_FULL_PATH} сохранён")
    print(f"{SVM_FULL_PATH} сохранён")
    print(f"{X_FULL_PATH} сохранён")
    print(f"{LABELS_FULL_PATH} сохранён")

print(f"\n[Total Time] {time.time() - t0:.2f} сек")
