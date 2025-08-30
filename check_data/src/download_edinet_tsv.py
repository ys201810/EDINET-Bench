import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import requests
import time
import zipfile
from tqdm import tqdm

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
    EDINET_API_KEY = os.getenv("EDINET_API_KEY")
    base_url = "https://api.edinet-fsa.go.jp/api/v2/documents/"
    get_type = 5
    data_path = Path(Path.cwd(), "check_data", "data")

    df_train_doc_id = pd.read_csv(Path(data_path, "train_doc_ids.tsv"), sep="\t")
    df_test_doc_id = pd.read_csv(Path(data_path, "test_doc_ids.tsv"), sep="\t")

    train_prev_doc_ids = df_train_doc_id["prev_year_doc_id"].tolist()
    train_current_doc_ids = df_train_doc_id["current_year_doc_id"].tolist()
    test_prev_doc_ids = df_test_doc_id["prev_year_doc_id"].tolist()
    test_current_doc_ids = df_test_doc_id["current_year_doc_id"].tolist()

    for doc_ids in [
        train_prev_doc_ids,
        train_current_doc_ids,
        test_prev_doc_ids,
        test_current_doc_ids,
    ]:
        for doc_id in tqdm(doc_ids):
            url = (
                f"{base_url}{doc_id}?type={get_type}&Subscription-Key={EDINET_API_KEY}"
            )
            response = fetch_url(url)

            # zipファイルの保存
            zip_file_path = data_path / "zip" / f"{doc_id}.zip"

            if response and response.content:
                with open(zip_file_path, "wb") as f:
                    f.write(response.content)
            else:
                continue

            # zipファイルの解凍
            csv_file_path = data_path / "csv" / f"{doc_id}"
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(csv_file_path)


if __name__ == "__main__":
    main()
