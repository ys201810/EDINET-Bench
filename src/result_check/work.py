import pandas as pd
from pathlib import Path


def check_test_data_industry():
    # edinetcode.csvの読み込み
    edinetcode_path = Path().cwd() / "data" / "edinetcode.csv"
    df = pd.read_csv(edinetcode_path)
    df = df[["ＥＤＩＮＥＴコード", "提出者名", "提出者業種", "資本金"]]
    
    


def main():
    # 予測結果の読み込み
    target = ""
    
    
    # テストデータの業種分類をチェックする
    check_test_data_industry()


if __name__ == "__main__":
    main()
