import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from ml_platform.data.uci_sms import load_uci_sms
from ml_platform.models.classifiers.nb import MultinomialNB
from ml_platform.tracking import log_metric, log_param, start_run

DATA_PATH = Path("ml_platform/data/raw/SMSSpamCollection")
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "baseline.json"
SEED = 42
EXPERIMENT_NAME = "sms-baseline"
ALPHAS = [0.1, 0.5, 1.0, 2.0]


def run_baseline(
    data_path: str | Path = DATA_PATH,
    report_path: str | Path = REPORT_PATH,
) -> dict:
    raw_data = load_uci_sms(data_path)
    data = raw_data.drop_duplicates(subset="text").reset_index(drop=True)

    train_data, remainder = train_test_split(
        data, test_size=0.30, random_state=SEED, stratify=data["label"]
    )

    validation_data, test_data = train_test_split(
        remainder,
        test_size=0.50,
        random_state=SEED,
        stratify=remainder["label"],
    )

    assert set(train_data["text"]).isdisjoint(validation_data["text"])
    assert set(train_data["text"]).isdisjoint(test_data["text"])
    assert set(validation_data["text"]).isdisjoint(test_data["text"])

    print("Исходных строк", len(raw_data))
    print("Уникальных строк", len(data))
    print("Размеры выборок: ", len(train_data), len(validation_data), len(test_data))
    print(data["label"].value_counts())

    for name, split in (
        ("train", train_data),
        ("validation", validation_data),
        ("test", test_data),
    ):
        print(name, len(split), split["label"].value_counts().to_dict())

    def evaluate(y_true, y_pred):
        # Считаем precision, recall и F1 для положительного класса spam.
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            pos_label="spam",
            average="binary",
            zero_division=0,
        )

        # Порядок строк и столбцов матрицы: ham, spam.
        matrix = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "spam_precision": float(precision),
            "spam_recall": float(recall),
            "spam_f1": float(f1),
            "confusion_matrix": matrix.tolist(),
        }

    best_alpha = None
    best_validation_f1 = -1.0

    for alpha in ALPHAS:
        run_name = f"multinomial-nb-alpha-{alpha:g}"
        with start_run(EXPERIMENT_NAME, run_name) as run:
            log_param("model", "MultinomialNB")
            log_param("alpha", alpha)
            log_param("seed", SEED)

            model = MultinomialNB(alpha=alpha)
            model.fit(train_data["text"].tolist(), train_data["label"].tolist())
            predictions = [model.predict(text) for text in validation_data["text"]]
            metrics = evaluate(validation_data["label"].tolist(), predictions)

            for key in ("accuracy", "spam_precision", "spam_recall", "spam_f1"):
                log_metric(f"validation_{key}", metrics[key])

            print(
                f"alpha={alpha}: "
                f"validation spam F1={metrics['spam_f1']:.4f}, "
                f"precision={metrics['spam_precision']:.4f}, "
                f"recall={metrics['spam_recall']:.4f}, "
                f"accuracy={metrics['accuracy']:.4f}"
            )
            print("Tracking run:", run.run_id)

        # Выбираем alpha по F1 на validation, не по test.
        if metrics["spam_f1"] > best_validation_f1:
            best_alpha = alpha
            best_validation_f1 = metrics["spam_f1"]

    train_validation = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    with start_run(EXPERIMENT_NAME, "multinomial-nb-final") as final_run:
        log_param("model", "MultinomialNB")
        log_param("alpha", best_alpha)
        log_param("seed", SEED)
        log_metric("validation_spam_f1", best_validation_f1)

        best_model = MultinomialNB(alpha=best_alpha)
        best_model.fit(
            train_validation["text"].tolist(),
            train_validation["label"].tolist(),
        )

        # Test используется только для итоговой оценки.
        test_predictions = [best_model.predict(text) for text in test_data["text"]]
        naive_bayes_test = evaluate(test_data["label"].tolist(), test_predictions)
        print("Naive Bayes test:", naive_bayes_test)

        majority_label = train_validation["label"].mode().iloc[0]
        majority_predictions = [majority_label] * len(test_data)
        majority_test = evaluate(test_data["label"].tolist(), majority_predictions)
        print("Majority baseline test:", majority_test)

        for model_name, metrics in (
            ("naive_bayes", naive_bayes_test),
            ("majority_baseline", majority_test),
        ):
            for key in ("accuracy", "spam_precision", "spam_recall", "spam_f1"):
                log_metric(f"test_{model_name}_{key}", metrics[key])

        print("Tracking run:", final_run.run_id)

    report = {
        "dataset": {
            "source_rows": len(raw_data),
            "unique_texts": len(data),
            "removed_duplicates": len(raw_data) - len(data),
        },
        "split": {
            "seed": SEED,
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "model_selection": {
            "model": "MultinomialNB",
            "alpha_candidates": ALPHAS,
            "selected_alpha": best_alpha,
            "validation_spam_f1": best_validation_f1,
        },
        "test": {
            "naive_bayes": naive_bayes_test,
            "majority_baseline": majority_test,
        },
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    run_baseline()


if __name__ == "__main__":
    main()
