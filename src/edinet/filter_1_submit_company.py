import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="複数回提出している企業を除外するフィルター"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="入力TSVファイルのパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力TSVファイルのパス",
    )
    parser.add_argument(
        "--edinet_code_csv",
        type=str,
        default="../edinetcode.csv",
        help="EDINETコードCSVファイルのパス",
    )
    args = parser.parse_args()

    # データ読み込み
    df = pd.read_csv(args.input, sep="\t")

    # EDINETコード情報の読み込み
    df_edinet = pd.read_csv(args.edinet_code_csv, sep=",")
    df_edinet = df_edinet.rename({"ＥＤＩＮＥＴコード": "edinet_code"}, axis=1)

    # マージ
    df = pd.merge(df, df_edinet[["edinet_code", "提出者業種"]], on="edinet_code")

    # 提出回数をカウント
    df_cnt = df.groupby("filer_name").size().reset_index().rename({0: "cnt"}, axis=1)

    # 2回以上提出している企業をリストアップ
    over_2_submit_companies = df_cnt.query("cnt > 1")["filer_name"].to_list()

    # フィルタリング前後の件数を表示
    print(f"フィルタリング前: {len(df)}")
    df = df.query("filer_name not in @over_2_submit_companies")
    print(f"フィルタリング後: {len(df)}")

    # 結果を保存
    df.to_csv(args.output, sep="\t", index=False)
    print(f"結果を保存しました: {args.output}")


if __name__ == "__main__":
    main()
