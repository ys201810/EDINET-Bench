"""
提出書類一覧のAPIで、提出書類情報を作成する。
doc_type_code == "120"を変更することで、取得する書類の種類を指定。
"""

from dotenv import load_dotenv
import requests
import json
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import os
import argparse

load_dotenv()


def fetch_url(url):
    with requests.Session() as session:
        try:
            response = requests.get(url)
            response.raise_for_status()  # エラー時は例外を発生
            time.sleep(0.001)
            return response

        except Exception as e:
            print(f"エラー発生: {e}")
            return ""


def make_url_parameter(date, type):
    return (
        os.getenv("DOC_LIST_BASE_URL")
        + f"?date={date}&type={type}&Subscription-Key={os.getenv('EDINET_API_KEY')}"
    )


def main(start_date_str, end_date_str, output_filename=None):
    # 対象日付の範囲作成
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    target_dates = []
    current_date = start_date

    while current_date <= end_date:
        target_dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    # 出力ファイル名を決定
    if output_filename:
        filename = output_filename
    else:
        start_date_formatted = start_date.strftime("%Y%m%d")
        end_date_formatted = end_date.strftime("%Y%m%d")
        filename = f"df_doc_type120_{start_date_formatted}_{end_date_formatted}.tsv"

    output_path = Path(Path().cwd() / "data" / "submitted_doc_ids" / filename)

    write_header = not Path(output_path).exists()
    error_dates = []

    for target_date in tqdm(target_dates):
        # 提出書類一覧取得
        url = make_url_parameter(target_date, "2")
        response = fetch_url(url)
        if response == "":
            print(f"{target_date}が失敗")
            error_dates.append(target_date)
            continue

        json_response = json.loads(response.text)

        # 結果取得
        results = json_response.get("results", [])

        # 日付単位の一時リスト
        daily_records = []

        for result in results:
            doc_type_code = result["docTypeCode"]
            if doc_type_code == "120":  # 有価証券報告書
                daily_records.append(
                    {
                        
                        "target_date": target_date,
                        "doc_id": result["docID"],
                        "doc_type_code": result["docTypeCode"],
                        "filer_name": result["filerName"],
                        "submit_date_time": result["submitDateTime"],
                        "doc_description": result["docDescription"],
                        "legal_status": result["legalStatus"],
                        "csv_flg": result["csvFlag"],
                        "edinet_code": result["edinetCode"],
                    }
                )

        # DataFrame化して追記保存
        if daily_records:
            daily_df = pd.DataFrame(daily_records)
            daily_df.to_csv(
                output_path, sep="\t", mode="a", index=False, header=write_header
            )
            write_header = False  # 最初だけヘッダーを書いたのでフラグ更新

        print(f"日付:{target_date} 件数:{len(daily_records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDINET提出書類一覧の取得")
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="取得開始日 (YYYY-MM-DD形式)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="取得終了日 (YYYY-MM-DD形式)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="出力ファイル名 (指定しない場合は df_doc_type120_{start_date}_{end_date}.tsv)",
    )
    args = parser.parse_args()

    main(args.start_date, args.end_date, args.output_filename)
