import argparse
import pandas as pd
import requests
import time
import json
from pathlib import Path
import zipfile
from tqdm import tqdm
import pickle
import os
from dotenv import load_dotenv

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


def main():
    parser = argparse.ArgumentParser(
        description="EDINET APIから文書をダウンロードしてZIPを解凍"
    )
    parser.add_argument(
        "--target_tsv_file_name",
        type=str,
        required=True,
        help="対象のTSVファイルのパス",
    )
    args = parser.parse_args()

    # CSVファイルの読み込み
    target_tsv_file = Path(args.target_tsv_file_name)
    df = pd.read_csv(target_tsv_file, sep="\t", encoding="utf-8")

    already_downloaded = []
    if Path("already_downloaded.pickle").exists():
        with open("already_downloaded.pickle", "rb") as f:
            already_downloaded = pickle.load(f)

    doc_ids = df.query("csv_flg == 1")["doc_id"]
    doc_ids = [doc_id for doc_id in doc_ids if doc_id not in already_downloaded]
    DOC_INFO_BASE_URL = os.getenv("DOC_INFO_BASE_URL")
    get_type = "5"

    for doc_id in tqdm(doc_ids):
        url = f"{DOC_INFO_BASE_URL}{doc_id}?type={get_type}&Subscription-Key={os.getenv('EDINET_API_KEY')}"
        response = fetch_url(url)

        # zipファイルの保存
        data_path = Path.cwd() / "data"
        zip_file_path = data_path / "zip" / f"{doc_id}.zip"

        if response and response.content:
            if not zip_file_path.parent.exists():
                zip_file_path.parent.mkdir(parents=True)
            with open(zip_file_path, "wb") as f:
                f.write(response.content)
        else:
            continue

        # zipファイルの解凍
        data_path = Path.cwd() / "data"
        csv_file_path = data_path / "csv" / f"{doc_id}"
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(csv_file_path)
        already_downloaded.append(doc_id)
        with open("already_downloaded.pickle", "wb") as f:
            pickle.dump(already_downloaded, f)


if __name__ == "__main__":
    main()
