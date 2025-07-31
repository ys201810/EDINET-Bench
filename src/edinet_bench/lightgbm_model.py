import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import re

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("LightGBM is not installed. Please install it with: pip install lightgbm")

try:
    from .utils import create_differential_features, create_percentage_change_features
except ImportError:
    from utils import create_differential_features, create_percentage_change_features

DATA_KEY = "summary"


def clean_feature_names(feature_names):
    """Clean feature names for LightGBM compatibility"""
    cleaned_names = []
    for name in feature_names:
        # Replace Japanese characters and special characters with safe alternatives
        cleaned = re.sub(r'[^\w\d_]', '_', name)
        # Remove consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        cleaned_names.append(cleaned)
    return cleaned_names


def prepare_dataset(task: str, split: str, use_differential_features=False, use_percentage_change_features=False):
    ds = load_dataset("SakanaAI/EDINET-Bench", task, split=split)
    doc_ids = ds["doc_id"]
    print(ds[0][DATA_KEY])  # デバッグ表示（初期データの確認）

    data_list = [
        {**json.loads(example[DATA_KEY]), "label": int(example["label"])}
        for example in ds
    ]

    return preprocess_data(data_list, use_differential_features, use_percentage_change_features), doc_ids


def preprocess_data(data_list, use_differential_features=False, use_percentage_change_features=False):
    rows = []
    for data in data_list:
        row = {}
        for key, values in data.items():
            if key == "label":
                row[key] = values
            elif values is not None:
                for year, val in values.items():
                    col_name = f"{key}_{year}"
                    row[col_name] = float(val) if val not in ["－", None] else np.nan
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if use_differential_features:
        df = create_differential_features(df)
    
    if use_percentage_change_features:
        df = create_percentage_change_features(df)
    
    return df


def fill_and_align_data(X_train, X_test):
    train_mean = X_train.mean(numeric_only=True)
    X_train.fillna(train_mean, inplace=True)
    X_test.fillna(train_mean, inplace=True)

    constant_cols = X_train.columns[X_train.nunique() <= 1]
    X_train.drop(columns=constant_cols, inplace=True)
    X_test.drop(columns=constant_cols, inplace=True, errors="ignore")

    X_test = X_test.reindex(columns=X_train.columns)
    return X_train, X_test


def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(
        f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}"
    )
    return acc, prec, rec, f1


def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.show()


def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.show()
    print(f"AUC: {auc:.3f}")


def save_predictions(path, y_pred, y_true, probs, test_doc_ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for doc_id, pred, label, prob in zip(test_doc_ids, y_pred, y_true, probs):
            json.dump(
                {
                    "doc_id": doc_id,
                    "prediction": int(pred),
                    "prob": float(prob),
                    "label": int(label),
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")


def show_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    print(feature_importance.to_latex(index=False, float_format="%.3f"))


def parse_args():
    parser = ArgumentParser(description="LightGBM for Profit Forecast")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["fraud_detection", "earnings_forecast"],
        help="Task name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of boosting iterations.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=-1,
        help="Maximum tree leaves for base learners (-1 means no limit).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Boosting learning rate.",
    )
    parser.add_argument(
        "--num_leaves",
        type=int,
        default=31,
        help="Maximum tree leaves for base learners.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Subsample ratio of the training instances.",
    )
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=1.0,
        help="Subsample ratio of columns when constructing each tree.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "--use_differential_features",
        action="store_true",
        help="Enable differential features (differences and growth rates between years).",
    )
    parser.add_argument(
        "--use_percentage_change_features",
        action="store_true",
        help="Enable percentage change features (growth rates between years).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # データ読み込み・前処理
    train, _ = prepare_dataset(args.task, split="train", use_differential_features=args.use_differential_features, use_percentage_change_features=args.use_percentage_change_features)
    test, test_doc_ids = prepare_dataset(args.task, split="test", use_differential_features=args.use_differential_features, use_percentage_change_features=args.use_percentage_change_features)

    X_train, y_train = train.drop(columns=["label"]), train["label"]
    X_test, y_test = test.drop(columns=["label"]), test["label"]

    X_train, X_test = fill_and_align_data(X_train, X_test)

    # Clean feature names for LightGBM compatibility
    cleaned_names = clean_feature_names(X_train.columns)
    X_train.columns = cleaned_names
    X_test.columns = cleaned_names
    feature_names = cleaned_names

    # LightGBMはスケーリング不要

    # モデル学習
    model = LGBMClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
        verbose=-1,  # Suppress training output
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 評価と出力
    evaluate_model(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)
    show_feature_importance(model, feature_names=feature_names)
    save_predictions(
        os.path.join(args.output_dir, args.task, "lightgbm", f"{DATA_KEY}.jsonl"),
        y_pred,
        y_test,
        y_pred_proba,
        test_doc_ids,
    )


if __name__ == "__main__":
    main()