import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# локальные пути
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = Path("/root/NLP")
DATA_PATH = Path("/root/NLP/data/annotation.csv")
SBERT_MODEL_PATH = Path("/root/NLP/sbert_large_nlu_ru")
OUTPUT_DIR = Path("/root/NLP/models/TF_IDF_SBERT")

TEST_SIZE = 0.30
RANDOM_STATE = 42
DEVICE = "cpu"
SBERT_BATCH_SIZE = 8   
CV_FOLDS = 3
CV_N_JOBS = 40

TFIDF_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 0.98,
    "max_features": 50000,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float32,
}

C_GRID = [1.0]
TFIDF_WEIGHTS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
SBERT_WEIGHTS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]

PREFERRED_LABEL_ORDER = ["anx", "dep", "neu", "pos"]


# очистка текста
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = " ".join(text.split())
    return text.strip()


def clean_label(label: str) -> str:
    if pd.isna(label):
        return ""
    label = str(label)
    label = label.replace("\r", " ").replace("\n", " ")
    label = label.replace('"', "")
    label = label.replace("\xa0", " ")
    label = label.replace("ё", "е").replace("Ё", "Е")
    label = " ".join(label.split())
    return label.strip()



def get_label_order(y_values):
    uniq = list(dict.fromkeys(map(str, y_values)))
    if set(uniq) == set(PREFERRED_LABEL_ORDER):
        return PREFERRED_LABEL_ORDER
    return sorted(set(uniq))



def compute_metrics(y_true, y_pred, label_order):
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        output_dict=True,
        digits=6,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)),
        "support_total": int(len(y_true)),
        "per_class": {},
    }

    for label in label_order:
        label_stats = report_dict.get(label, {})
        metrics["per_class"][label] = {
            "precision": float(label_stats.get("precision", 0.0)),
            "recall": float(label_stats.get("recall", 0.0)),
            "f1_score": float(label_stats.get("f1-score", 0.0)),
            "support": int(label_stats.get("support", 0)),
        }

    return metrics, report_dict



def print_metrics(y_true, y_pred, label_order, title="METRICS"):
    metrics, report_dict = compute_metrics(y_true, y_pred, label_order)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Accuracy           : {metrics['accuracy']:.6f}")
    print(f"Balanced accuracy  : {metrics['balanced_accuracy']:.6f}")
    print(f"MCC                : {metrics['mcc']:.6f}")
    print(f"Precision macro    : {metrics['precision_macro']:.6f}")
    print(f"Precision weighted : {metrics['precision_weighted']:.6f}")
    print(f"Recall macro       : {metrics['recall_macro']:.6f}")
    print(f"Recall weighted    : {metrics['recall_weighted']:.6f}")
    print(f"F1 macro           : {metrics['f1_macro']:.6f}")
    print(f"F1 weighted        : {metrics['f1_weighted']:.6f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=label_order,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=label_order))
    print("=" * 80 + "\n")

    return metrics, report_dict



def save_metrics_artifacts(y_true, y_pred, label_order, output_dir, prefix="test"):
    metrics, report_dict = compute_metrics(y_true, y_pred, label_order)

    metrics_json_path = output_dir / f"{prefix}_metrics.json"
    report_csv_path = output_dir / f"{prefix}_classification_report.csv"
    confusion_csv_path = output_dir / f"{prefix}_confusion_matrix.csv"
    predictions_csv_path = output_dir / f"{prefix}_predictions.csv"

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
    cm_df.index.name = "true_label"
    cm_df.to_csv(confusion_csv_path, encoding="utf-8-sig")

    report_rows = []
    for key, value in report_dict.items():
        if isinstance(value, dict):
            row = {"label": key}
            row.update(value)
            report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "true_label": list(y_true),
        "pred_label": list(y_pred),
    })
    pred_df.to_csv(predictions_csv_path, index=False, encoding="utf-8-sig")

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {
        "metrics_json": metrics_json_path,
        "classification_report_csv": report_csv_path,
        "confusion_matrix_csv": confusion_csv_path,
        "predictions_csv": predictions_csv_path,
    }, metrics



def encode_sbert(model, texts, batch_size=64):
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)



def build_hybrid_matrix(X_tfidf, X_sbert, tfidf_weight=1.0, sbert_weight=1.0):
    X_tfidf = X_tfidf.tocsr().astype(np.float32)
    X_sbert_sparse = csr_matrix((X_sbert * np.float32(sbert_weight)).astype(np.float32))
    X_hybrid = hstack(
        [X_tfidf * np.float32(tfidf_weight), X_sbert_sparse],
        format="csr",
        dtype=np.float32,
    )
    return X_hybrid


def main():
    total_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Загрузка корпуса...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("В annotation.csv должны быть колонки: text, label")

    df = df[["text", "label"]].copy()
    raw_size = len(df)

    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].map(clean_label)

    empty_text_rows = df[df["text"].str.len() == 0].copy()
    empty_label_rows = df[df["label"].str.len() == 0].copy()

    df = df[(df["text"].str.len() > 0) & (df["label"].str.len() > 0)].copy()
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("После очистки не осталось ни одной валидной строки.")

    texts = df["text"].tolist()
    labels = df["label"].tolist()
    label_order = get_label_order(labels)

    print(f"Исходный размер корпуса          : {raw_size}")
    print(f"Удалено пустых текстов           : {len(empty_text_rows)}")
    print(f"Удалено пустых меток             : {len(empty_label_rows)}")
    print(f"Размер корпуса после очистки     : {len(df)}")

    if len(empty_text_rows) > 0:
        empty_texts_path = OUTPUT_DIR / "dropped_empty_texts.csv"
        empty_text_rows.to_csv(empty_texts_path, index=False, encoding="utf-8-sig")
        print(f"Список пустых текстов сохранен: {empty_texts_path}")

    if len(empty_label_rows) > 0:
        empty_labels_path = OUTPUT_DIR / "dropped_empty_labels.csv"
        empty_label_rows.to_csv(empty_labels_path, index=False, encoding="utf-8-sig")
        print(f"Список пустых меток сохранен: {empty_labels_path}")

    print("\nРаспределение классов после очистки:")
    print(df["label"].value_counts())

    print("\n[2] Деление на train/test...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Train size: {len(X_train_text)}")
    print(f"Test size : {len(X_test_text)}")

    print("\n[3] Построение TF-IDF...")
    tfidf_start = time.time()

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_tfidf_train = vectorizer.fit_transform(X_train_text)
    X_tfidf_test = vectorizer.transform(X_test_text)

    print(f"TF-IDF train shape: {X_tfidf_train.shape}")
    print(f"TF-IDF test shape : {X_tfidf_test.shape}")
    print(f"[Time] TF-IDF: {time.time() - tfidf_start:.2f} сек")

    print("\n[4] Загрузка SBERT...")
    if not SBERT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена папка модели SBERT: {SBERT_MODEL_PATH}")

    sbert_load_start = time.time()
    sbert_model = SentenceTransformer(str(SBERT_MODEL_PATH), device=DEVICE)
    print(f"[Time] load SBERT: {time.time() - sbert_load_start:.2f} сек")

    print("\n[5] Построение SBERT-эмбеддингов для train...")
    sbert_train_start = time.time()
    X_sbert_train = encode_sbert(sbert_model, X_train_text, batch_size=SBERT_BATCH_SIZE)
    print(f"SBERT train shape: {X_sbert_train.shape}")
    print(f"[Time] SBERT train: {time.time() - sbert_train_start:.2f} сек")

    print("\n[6] Построение SBERT-эмбеддингов для test...")
    sbert_test_start = time.time()
    X_sbert_test = encode_sbert(sbert_model, X_test_text, batch_size=SBERT_BATCH_SIZE)
    print(f"SBERT test shape: {X_sbert_test.shape}")
    print(f"[Time] SBERT test: {time.time() - sbert_test_start:.2f} сек")

    print("\n[7] Подбор лучших весов и C через CV...")
    search_start = time.time()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search_results = []

    combo_num = 0
    total_combos = len(TFIDF_WEIGHTS) * len(SBERT_WEIGHTS) * len(C_GRID)

    for tfidf_w in TFIDF_WEIGHTS:
        for sbert_w in SBERT_WEIGHTS:
            X_hybrid_train = build_hybrid_matrix(
                X_tfidf_train,
                X_sbert_train,
                tfidf_weight=tfidf_w,
                sbert_weight=sbert_w,
            )
            X_hybrid_test = build_hybrid_matrix(
                X_tfidf_test,
                X_sbert_test,
                tfidf_weight=tfidf_w,
                sbert_weight=sbert_w,
            )

            for C_value in C_GRID:
                combo_num += 1
                print(
                    f"[{combo_num}/{total_combos}] "
                    f"tfidf_w={tfidf_w}, sbert_w={sbert_w}, C={C_value}"
                )

                clf = LinearSVC(
                    C=C_value,
                    class_weight="balanced",
                    loss="squared_hinge",
                    max_iter=200000,
                    random_state=RANDOM_STATE,
                )

                scores = cross_val_score(
                    clf,
                    X_hybrid_train,
                    y_train,
                    cv=cv,
                    scoring="f1_macro",
                    n_jobs=CV_N_JOBS,
                )

                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))

                clf.fit(X_hybrid_train, y_train)
                y_pred_combo = clf.predict(X_hybrid_test)
                combo_metrics, _ = compute_metrics(y_test, y_pred_combo, label_order)

                print(
                    "    "
                    f"CV f1_macro = {mean_score:.6f} ± {std_score:.6f} | "
                    f"TEST Acc = {combo_metrics['accuracy']:.6f} | "
                    f"TEST MCC = {combo_metrics['mcc']:.6f} | "
                    f"TEST Prec_w = {combo_metrics['precision_weighted']:.6f} | "
                    f"TEST Recall_w = {combo_metrics['recall_weighted']:.6f} | "
                    f"TEST F1_w = {combo_metrics['f1_weighted']:.6f}"
                )

                search_results.append(
                    {
                        "tfidf_weight": tfidf_w,
                        "sbert_weight": sbert_w,
                        "C": C_value,
                        "cv_f1_macro_mean": mean_score,
                        "cv_f1_macro_std": std_score,
                        "test_accuracy": combo_metrics["accuracy"],
                        "test_mcc": combo_metrics["mcc"],
                        "test_precision_weighted": combo_metrics["precision_weighted"],
                        "test_recall_weighted": combo_metrics["recall_weighted"],
                        "test_f1_weighted": combo_metrics["f1_weighted"],
                    }
                )

    results_df = pd.DataFrame(search_results).sort_values(
        by="cv_f1_macro_mean", ascending=False
    ).reset_index(drop=True)

    results_csv_path = OUTPUT_DIR / "hybrid_search_results.csv"
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")

    print("\nTOP-10 комбинаций по CV F1_macro:")
    print(
        results_df[
            [
                "tfidf_weight",
                "sbert_weight",
                "C",
                "cv_f1_macro_mean",
                "cv_f1_macro_std",
                "test_accuracy",
                "test_mcc",
                "test_precision_weighted",
                "test_recall_weighted",
                "test_f1_weighted",
            ]
        ].head(10).to_string(index=False)
    )

    best_row = results_df.iloc[0]
    best_tfidf_weight = float(best_row["tfidf_weight"])
    best_sbert_weight = float(best_row["sbert_weight"])
    best_C = float(best_row["C"])
    best_cv_score = float(best_row["cv_f1_macro_mean"])

    print("\nЛучшие параметры:")
    print(best_row)
    print(f"[Time] search: {time.time() - search_start:.2f} сек")

    print("\n[8] Обучение лучшей модели на train...")
    fit_start = time.time()

    X_hybrid_train_best = build_hybrid_matrix(
        X_tfidf_train,
        X_sbert_train,
        tfidf_weight=best_tfidf_weight,
        sbert_weight=best_sbert_weight,
    )
    X_hybrid_test_best = build_hybrid_matrix(
        X_tfidf_test,
        X_sbert_test,
        tfidf_weight=best_tfidf_weight,
        sbert_weight=best_sbert_weight,
    )

    best_clf = LinearSVC(
        C=best_C,
        class_weight="balanced",
        loss="squared_hinge",
        max_iter=200000,
        random_state=RANDOM_STATE,
    )
    best_clf.fit(X_hybrid_train_best, y_train)

    y_pred = best_clf.predict(X_hybrid_test_best)

    print(f"[Time] fit + predict: {time.time() - fit_start:.2f} сек")
    test_metrics, _ = print_metrics(
        y_test,
        y_pred,
        label_order,
        title="HYBRID TF-IDF + SBERT + LinearSVC (70/30)",
    )

    saved_metric_paths, saved_metrics = save_metrics_artifacts(
        y_test,
        y_pred,
        label_order,
        OUTPUT_DIR,
        prefix="test",
    )

    print("Сохранены файлы с метриками:")
    for key, path in saved_metric_paths.items():
        print(f" - {key}: {path}")

    print("\n[9] Переобучение лучшей конфигурации на 100% корпуса...")
    refit_start = time.time()

    vectorizer_full = TfidfVectorizer(**TFIDF_PARAMS)
    X_tfidf_full = vectorizer_full.fit_transform(texts)

    print("Построение SBERT-эмбеддингов для всего корпуса...")
    X_sbert_full = encode_sbert(sbert_model, texts, batch_size=SBERT_BATCH_SIZE)

    X_hybrid_full = build_hybrid_matrix(
        X_tfidf_full,
        X_sbert_full,
        tfidf_weight=best_tfidf_weight,
        sbert_weight=best_sbert_weight,
    )

    final_clf = LinearSVC(
        C=best_C,
        class_weight="balanced",
        loss="squared_hinge",
        max_iter=200000,
        random_state=RANDOM_STATE,
    )
    final_clf.fit(X_hybrid_full, labels)

    print(f"[Time] refit full: {time.time() - refit_start:.2f} сек")

    print("\n[10] Сохранение артефактов...")
    vectorizer_path = OUTPUT_DIR / "tfidf_vectorizer_hybrid.pkl"
    classifier_path = OUTPUT_DIR / "svm_hybrid_tfidf_sbert.pkl"
    meta_path = OUTPUT_DIR / "hybrid_meta.json"

    joblib.dump(vectorizer_full, vectorizer_path)
    joblib.dump(final_clf, classifier_path)

    meta = {
        "base_dir": str(BASE_DIR),
        "data_path": str(DATA_PATH),
        "sbert_model_path": str(SBERT_MODEL_PATH),
        "output_dir": str(OUTPUT_DIR),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "device": DEVICE,
        "sbert_batch_size": SBERT_BATCH_SIZE,
        "cv_folds": CV_FOLDS,
        "best_params": {
            "tfidf_weight": best_tfidf_weight,
            "sbert_weight": best_sbert_weight,
            "C": best_C,
            "cv_f1_macro_mean": best_cv_score,
        },
        "tfidf_params": {
            "lowercase": TFIDF_PARAMS["lowercase"],
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
            "min_df": TFIDF_PARAMS["min_df"],
            "max_df": TFIDF_PARAMS["max_df"],
            "max_features": TFIDF_PARAMS["max_features"],
            "sublinear_tf": TFIDF_PARAMS["sublinear_tf"],
            "norm": TFIDF_PARAMS["norm"],
        },
        "label_order": label_order,
        "raw_size": raw_size,
        "n_samples_full_after_cleaning": len(texts),
        "dropped_empty_texts": int(len(empty_text_rows)),
        "dropped_empty_labels": int(len(empty_label_rows)),
        "test_metrics": saved_metrics,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nСохранено:")
    print(f" - {vectorizer_path}")
    print(f" - {classifier_path}")
    print(f" - {meta_path}")
    print(f" - {results_csv_path}")
    for path in saved_metric_paths.values():
        print(f" - {path}")

    print(f"\n[TOTAL TIME] {time.time() - total_start:.2f} сек")


if __name__ == "__main__":
    main()
