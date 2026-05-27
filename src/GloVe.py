import os
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import torch
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
from joblib import dump

# настройка CPU-потоков до импорта numpy/torch
N_THREADS = 48
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS)

# настройки PyTorch CPU
torch.set_num_threads(N_THREADS)
try:
    torch.set_num_interop_threads(N_THREADS)
except RuntimeError:
    pass

torch.manual_seed(42)

os.makedirs("/root/NLP/models/GloVe", exist_ok=True)

GLOVE_ITERS = 50
GLOVE_BATCH_SIZE = 131072

# построение co-occurrence (48 процессов)
def _build_cooc_chunk(args):
    chunk, window = args
    local_cooc = defaultdict(float)

    for tokens in chunk:
        for i, w in enumerate(tokens):
            L = max(0, i - window)
            R = min(len(tokens), i + window + 1)
            for j in range(L, R):
                if i == j:
                    continue
                dist = abs(i - j)
                local_cooc[(w, tokens[j])] += 1.0 / dist

    return local_cooc


def build_cooc(corpus, window=6, n_workers=N_THREADS):
    if n_workers <= 1 or len(corpus) < 2:
        return _build_cooc_chunk((corpus, window))

    chunk_size = max(1, len(corpus) // n_workers)
    chunks = [corpus[i:i + chunk_size] for i in range(0, len(corpus), chunk_size)]

    with Pool(processes=n_workers) as pool:
        partials = pool.map(_build_cooc_chunk, [(chunk, window) for chunk in chunks])

    cooc = defaultdict(float)
    for part in partials:
        for k, v in part.items():
            cooc[k] += v

    return cooc

# ускоренное GloVe через batched PyTorch CPU
def glove_train(cooc, vocab, dim=300, iters=GLOVE_ITERS, x_max=100.0, alpha=0.75, lr=0.05,
                batch_size=GLOVE_BATCH_SIZE):
    vocab_list = list(vocab)
    word2id = {w: i for i, w in enumerate(vocab_list)}
    V = len(vocab_list)

    # преобразуем словарь co-occurrence в плотные массивы индексов/значений.
    i_idx = []
    j_idx = []
    x_vals = []
    for (w, c), x in cooc.items():
        if w in word2id and c in word2id:
            i_idx.append(word2id[w])
            j_idx.append(word2id[c])
            x_vals.append(x)

    i_idx = np.asarray(i_idx, dtype=np.int64)
    j_idx = np.asarray(j_idx, dtype=np.int64)
    x_vals = np.asarray(x_vals, dtype=np.float32)

    weights = np.minimum((x_vals / x_max) ** alpha, 1.0).astype(np.float32)
    log_x = np.log(x_vals).astype(np.float32)

    device = torch.device("cpu")

    i_t = torch.from_numpy(i_idx).to(device)
    j_t = torch.from_numpy(j_idx).to(device)
    w_t = torch.from_numpy(weights).to(device)
    log_x_t = torch.from_numpy(log_x).to(device)

    W = torch.nn.Embedding(V, dim, device=device)
    C = torch.nn.Embedding(V, dim, device=device)
    bW = torch.nn.Embedding(V, 1, device=device)
    bC = torch.nn.Embedding(V, 1, device=device)

    with torch.no_grad():
        W.weight.normal_(mean=0.0, std=0.01)
        C.weight.normal_(mean=0.0, std=0.01)
        bW.weight.zero_()
        bC.weight.zero_()

    optimizer = torch.optim.Adagrad(
        [W.weight, C.weight, bW.weight, bC.weight],
        lr=lr
    )

    n_pairs = i_t.shape[0]
    print(f"GloVe pairs: {n_pairs:,} | vocab: {V:,} | dim: {dim} | threads: {N_THREADS}")

    for epoch in range(iters):
        t_epoch = time.time()
        perm = torch.randperm(n_pairs, device=device)
        total_loss = 0.0
        seen = 0

        for start in range(0, n_pairs, batch_size):
            idx = perm[start:start + batch_size]

            bi = i_t[idx]
            bj = j_t[idx]
            bw = w_t[idx]
            blogx = log_x_t[idx]

            wi = W(bi)
            cj = C(bj)
            bias_i = bW(bi).squeeze(1)
            bias_j = bC(bj).squeeze(1)

            scores = (wi * cj).sum(dim=1) + bias_i + bias_j
            loss_vec = bw * (scores - blogx).pow(2)
            loss = loss_vec.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = idx.numel()
            total_loss += loss.item() * batch_n
            seen += batch_n

        avg_loss = total_loss / max(seen, 1)
        print(f"[Epoch {epoch + 1}/{iters}] loss = {avg_loss:.6f} | time = {time.time() - t_epoch:.2f} сек")

    with torch.no_grad():
        emb = (W.weight + C.weight).cpu().numpy().astype(np.float32)

    return emb, word2id


# GloVe-эмбеддинги
def sent_emb(tokens, emb, word2id, dim):
    vecs = [emb[word2id[t]] for t in tokens if t in word2id]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32)


def _sent_emb_chunk(tokens_chunk, emb, word2id, dim):
    return np.vstack([sent_emb(tokens, emb, word2id, dim) for tokens in tokens_chunk])


def build_sentence_matrix(tokens_list, emb, word2id, dim, n_workers=N_THREADS):
    if n_workers <= 1 or len(tokens_list) < 2:
        return np.vstack([sent_emb(t, emb, word2id, dim) for t in tokens_list])

    chunk_size = max(1, len(tokens_list) // n_workers)
    chunks = [tokens_list[i:i + chunk_size] for i in range(0, len(tokens_list), chunk_size)]

    # используем threads, чтобы не копировать emb в 48 отдельных процессов.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        parts = list(executor.map(lambda chunk: _sent_emb_chunk(chunk, emb, word2id, dim), chunks))

    return np.vstack(parts)

# загрузка данных
DATA = "/root/NLP/data/annotation.csv"
df = pd.read_csv(DATA)

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

texts = df.text.tolist()
labels = df.label.tolist()

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=50, stratify=labels
)

train_tokens = [simple_preprocess(t) for t in X_train_texts]
test_tokens = [simple_preprocess(t) for t in X_test_texts]

print("\n=== Строим train co-occurrence ===")
cooc = build_cooc(train_tokens, window=4, n_workers=N_THREADS)
vocab = set(w for toks in train_tokens for w in toks)

# размерность эмбеддингов и штраф ошибки SVC
dims = [50, 100, 150, 200, 300]
Cs = [0.1, 0.3, 0.5, 1.0]

results = []

for dim in dims:
    print(f"\n=== dim={dim} ===")

    emb, word2id = glove_train(cooc, vocab, dim=dim, iters=GLOVE_ITERS)

    X_train = build_sentence_matrix(train_tokens, emb, word2id, dim, n_workers=N_THREADS)
    X_test = build_sentence_matrix(test_tokens, emb, word2id, dim, n_workers=N_THREADS)

    for C in Cs:
        clf = LinearSVC(C=C, max_iter=200000, loss="squared_hinge")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        results.append((dim, C, f1, acc))

        print(f"    C={C} → F1={f1:.4f}, Acc={acc:.4f}")

# лучшая модель
best = max(results, key=lambda x: x[2])
best_dim, best_C, best_f1, best_acc = best

print("\n=== ЛУЧШАЯ МОДЕЛЬ ===")
print("dim =", best_dim)
print("C   =", best_C)
print("F1  =", best_f1)

# финальное обучение на 0.7
print("\n=== Финальное обучение на train 70% ===")

emb, word2id = glove_train(cooc, vocab, dim=best_dim, iters=GLOVE_ITERS)

X_train = build_sentence_matrix(train_tokens, emb, word2id, best_dim, n_workers=N_THREADS)
X_test = build_sentence_matrix(test_tokens, emb, word2id, best_dim, n_workers=N_THREADS)

clf = LinearSVC(C=best_C, max_iter=200000, loss="squared_hinge")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# метрики 
print("\n=== МЕТРИКИ (train 70%) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("MCC:", matthews_corrcoef(y_test, y_pred))

print("\n--- WEIGHTED ---")
print("Precision_w:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
print("Recall_w:   ", recall_score(y_test, y_pred, average="weighted", zero_division=0))
print("F1_w:       ", f1_score(y_test, y_pred, average="weighted", zero_division=0))

print("\n--- MACRO ---")
print("Precision_macro:", precision_score(y_test, y_pred, average="macro", zero_division=0))
print("Recall_macro:   ", recall_score(y_test, y_pred, average="macro", zero_division=0))
print("F1_macro:       ", f1_score(y_test, y_pred, average="macro", zero_division=0))

print("\n--- MICRO ---")
print("Precision_micro:", precision_score(y_test, y_pred, average="micro", zero_division=0))
print("Recall_micro:   ", recall_score(y_test, y_pred, average="micro", zero_division=0))
print("F1_micro:       ", f1_score(y_test, y_pred, average="micro", zero_division=0))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# финальное дообучение на 100% данных
print("\n=== ДОобучение на 100% данных ===")

full_tokens = [simple_preprocess(t) for t in texts]
full_labels = labels

print("\n=== Строим full co-occurrence ===")
full_cooc = build_cooc(full_tokens, window=4, n_workers=N_THREADS)
full_vocab = set(w for toks in full_tokens for w in toks)

emb_full, word2id_full = glove_train(full_cooc, full_vocab, dim=best_dim, iters=GLOVE_ITERS)

X_full = build_sentence_matrix(full_tokens, emb_full, word2id_full, best_dim, n_workers=N_THREADS)

clf_full = LinearSVC(C=best_C, max_iter=200000, loss="squared_hinge")
clf_full.fit(X_full, full_labels)

# сохранение
np.save("/root/NLP/models/GloVe/glove_full_embeddings.npy", emb_full)
dump(word2id_full, "/root/NLP/models/GloVe/glove_full_word2id.pkl")
dump(clf_full, "/root/NLP/models/GloVe/svm_glove_full.pkl")

print("\n=== FINISHED: модель обучена на 100% датасета ===")
print("/root/NLP/models/GloVe/svm_glove_full.pkl")
print("/root/NLP/models/GloVe/glove_full_embeddings.npy")
print("/root/NLP/models/GloVe/glove_full_word2id.pkl")
