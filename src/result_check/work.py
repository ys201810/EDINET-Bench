import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    matthews_corrcoef,
)

mapping_33_to_17_ja = {
    # Foods
    "水産・農林業": "食品",
    "食料品": "食品",
    # Energy Resources
    "鉱業": "エネルギー資源",
    "石油・石炭製品": "エネルギー資源",
    # Construction & Materials
    "建設業": "建設・資材",
    "金属製品": "建設・資材",
    "ガラス・土石製品": "建設・資材",
    # Raw Materials & Chemicals
    "繊維製品": "建設・資材",
    "パルプ・紙": "原材料・化学",
    "化学": "原材料・化学",
    # Pharmaceutical
    "医薬品": "医薬品",
    # Automobiles & Transportation Equipment
    "ゴム製品": "自動車・輸送機",
    "輸送用機器": "自動車・輸送機",
    # Steel & Nonferrous Metals
    "鉄鋼": "鉄鋼・非鉄金属",
    "非鉄金属": "鉄鋼・非鉄金属",
    # Machinery
    "機械": "機械",
    # Electric Appliances & Precision Instruments
    "電気機器": "電気機器・精密機器",
    "精密機器": "電気機器・精密機器",
    "その他製品": "電気機器・精密機器",
    # IT & Services, Others
    "情報・通信業": "IT・サービス他",
    "サービス業": "IT・サービス他",
    # Electric Power & Gas
    "電気・ガス業": "電力・ガス",
    # Transportation & Logistics
    "陸運業": "交通・物流",
    "海運業": "交通・物流",
    "空運業": "交通・物流",
    "倉庫・運輸関連": "交通・物流",
    # Retail Trade
    "小売業": "小売",
    # Commercial & Wholesale Trade
    "卸売業": "商社・卸売",
    # Bank
    "銀行業": "銀行",
    # Financials (ex Banks)
    "証券、商品先物取引業": "金融（銀行除く）",
    "保険業": "金融（銀行除く）",
    "その他金融業": "金融（銀行除く）",
    # Real Estate
    "不動産業": "不動産",
}


def calc_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    if len(set(y_true)) < 2:
        # 片方のクラスしか存在しない場合、precision, recall, f1, mcc, roc_aucは計算できないのでNaNを返す
        return (
            accuracy,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            [],
            [],
        )
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    return accuracy, precision, recall, f1, mcc, roc_auc, fpr, tpr


def check_test_data_industry(df_result):
    # edinetcode.csvの読み込み
    edinetcode_path = Path().cwd() / "data" / "edinetcode.csv"
    df = pd.read_csv(edinetcode_path)
    df = df[["ＥＤＩＮＥＴコード", "提出者名", "提出者業種", "資本金"]]
    df.rename(
        columns={
            "ＥＤＩＮＥＴコード": "edinet_code",
            "提出者名": "company_name",
            "提出者業種": "industry",
            "資本金": "capital",
        },
        inplace=True,
    )

    df_result = pd.merge(df_result, df, on="edinet_code")
    df_result["industry_17"] = df_result["industry"].map(mapping_33_to_17_ja)

    # 全体での評価指標を計算
    accuracy, precision, recall, f1, mcc, roc_auc, fpr, tpr = calc_metrics(
        df_result["gt_label"], df_result["pred_label"], df_result["prob"]
    )
    print(f"全体: 件数:{len(df_result)} 評価指標(ROC-AUC): {roc_auc:.4f}")

    for industry_17 in df_result["industry_17"].unique():
        df_tmp = df_result[df_result["industry_17"] == industry_17]
        if len(df_tmp) == 0:
            continue
        accuracy, precision, recall, f1, mcc, roc_auc, fpr, tpr = calc_metrics(
            df_tmp["gt_label"], df_tmp["pred_label"], df_tmp["prob"]
        )
        print(
            f"業種: {industry_17}, 件数:{len(df_tmp)} 評価指標(ROC-AUC): {roc_auc:.4f}"
        )


def main():
    # 予測結果の読み込み
    base_path = Path().cwd()
    target = "vertex-ai/claude-3-7-sonnet-20250219"  # "o4-mini-2025-04-16"
    result_file = Path(
        base_path,
        "result",
        "earnings_forecast",
        target,
        "bs_cf_pl_summary_イジった.jsonl",
    )

    edinet_codes, doc_ids, gt_labels, probs, pred_labels = [], [], [], [], []
    with open(result_file, "r") as inf:
        for line in inf:
            result_dict = json.loads(line)
            edinet_codes.append(result_dict["edinet_code"])
            doc_ids.append(result_dict["doc_id"])
            gt_labels.append(result_dict["label"])
            probs.append(result_dict["prob"])
            pred_labels.append(result_dict["prediction"])

    df_results = pd.DataFrame(
        {
            "edinet_code": edinet_codes,
            "doc_id": doc_ids,
            "gt_label": gt_labels,
            "prob": probs,
            "pred_label": pred_labels,
        }
    )

    df_pred_nan = df_results[df_results["pred_label"].isna()]
    print(f"予測がNaNの件数: {len(df_pred_nan)}")
    df_result = df_results[~df_results["pred_label"].isna()]
    df_result["pred_label"] = df_result["pred_label"].astype(int)

    # テストデータの業種分類をチェックする
    check_test_data_industry(df_result)


if __name__ == "__main__":
    main()
