#　paper_work
このディレトリは、sigfin35に提出した論文の作業メモです。  

## 前提
- かなり速度優先で作業をしたため、他者環境での再現などは一切考慮していない。
- そのためコード重複等がかなりあり、もし続きで何か作業をする場合は、整理して別ブランチなどを切って作業することを強く推奨。
- 利益の増減予測(earning forecasting)のみを動作確認しています。

## 実施内容
1. API設定
2. EDINET-BENCHのオリジナルlogistic回帰での予測
3. EDINET-BENCHのオリジナルLLMの予測
4. EDINET-BENCHのオリジナルテストデータのLLM予測に対する業種/売上規模別の精度評価
5. EDINET-BENCH利用外の任意データの収集
6. EDINET-BENCH利用外データを使ったLLMの予測
7. EDINET-BENCH利用外データ予測に対する業種/売上規模別の精度評価

## 詳細メモ
### 1. API設定
EDINET-BENCHでは、EDINET-APIやOPENAI-APIなどのAPIを利用しています。  
これらのAPI-KEYは.envで設定しているため、.envを作成してください。

```
$ cd /path/to/EDINET-BENCH
$ cp .env_sample .env
# .envの各値に適切な値をセットしてください。
```

### 2. EDINET-BENCHのオリジナルlogistic回帰での予測
オリジナル論文のEDINET-BENCHのlogistic回帰を実行して、精度を確認します。  

```
# python環境構築
$ uv sync
$ source .venv/bin/activate

$ python src/edinet_bench/logistic.py --task earnings_forecast 
```

上記実行により、標準出力でAUC: 0.561が確認できます。  
予測結果は、result/earnings_forecast/logistic/summary.jsonlに出力されます。  
.vscode/launch.jsonの「Debug - earnings_forecast_logistic」に当たります。  

### 3. EDINET-BENCHのオリジナルLLMの予測
オリジナル論文のEDINET-BENCHのLLMでの予測を実行して、精度を確認します。  

```
$ python src/edinet_bench/predict.py --task earnings_forecast --model gpt-4o-2024-11-20 --sheets bs cf pl summary
```

上記実行により、標準出力でAUC等が確認できます。  
--modelで指定する利用可能なモデル名は以下の辞書のkeyです。  

```
# src/edinet_bench/model.pyのMODEL_TABLE変数から抜粋
MODEL_TABLE: dict[str, Model] = {
    "claude-3-5-sonnet-20241022": AnthropicModel,
    "claude-3-7-sonnet-20250219": AnthropicModel,
    "claude-3-5-haiku-20241022": AnthropicModel,
    "gpt-4o-2024-11-20": OpenAIModel,
    "o4-mini-2025-04-16": OpenAIModel,
    "deepseek/deepseek-r1": OpenRouterModel,
    "deepseek/deepseek-chat": OpenRouterModel,
    "vertex-ai/claude-3-7-sonnet-20250219": VertexAIModel,
    "vertex-ai/claude-3-5-sonnet-20241022": VertexAIModel,
}
```

オリジナルのコードからvertex-ai経由でのclaude利用が増えています。  

### 4. EDINET-BENCHのオリジナルテストデータのLLM予測に対する業種/売上規模別の精度評価
EDINET-BENCHのテストデータをLLMで予測した結果に対し、業種/売上規模別に評価します。  

```
# 業種別の精度確認
$ python src/result_check/work.py --model vertex-ai/claude-3-7-sonnet-20250219 --result_file_name bs_cf_pl_summary_text_roc0607.jsonl

# 売上規模別の精度確認
$ python test_result/earnings_forecast/logistic/test_sales_check.py
# ファイル内でllmの実行結果のファイルを指定しているため、ここを書き換えて実行する必要があります。
```

売上規模別の実行例

```
売上 50-75% の件数:95 ROC-AUC: 0.7048 増加件数:60
売上 下位25% の件数:96 ROC-AUC: 0.5330 増加件数:65
売上 上位25% の件数:96 ROC-AUC: 0.6729 増加件数:67
売上 25-50% の件数:95 ROC-AUC: 0.5852 増加件数:56
```

### 5. EDINET-BENCH利用外の任意データの収集
任意データを推論するためのデータを作成します。推論と結果確認には、前年の有価証券報告書と当年の有価証券報告書が必要のため、EDINETコード（企業コード）ごとにこの二つのペアを作成します。  
edinet_toyという白井作成の別ブランチで実施しました。  
こちらは雑多メモだけ残します。  

```
1. edinetから提出された年次有価証券報告書の一覧を取得
src/get_all_submit.pyのstart_dateとend_dateを、2025-06-01と2025-08-31に指定して実行。

2. 1の結果のファイルを使ってtsvを取得
src/get_csvs.pyを実行。target_tsv_fileを1の出力ファイル名に変更して実行。
exp_asset/20250601以降のみ/filter_1_submit_company.pyで複数提出がある会社を対象外にしてから実行。

# 以下は、edinet-benchで予測するためには2年分が必要なため、前年分を取得するために実施。
3. edinetから提出された年次有価証券報告書の一覧を取得
src/get_all_submit.pyのstart_dateとend_dateを、2025-04-01と2025-05-31に指定して実行。

4. 1の結果のファイルを使ってtsvを取得
src/get_csvs.pyを実行。target_tsv_fileを1の出力ファイル名に変更して実行。

5. 前年と当年のペアディレクトリを作成
exp_asset/make_pair.pyを実施。
実施後、exp_asset/own/にEDINETコードごとのペアディレクトリができるため、これをedinet-benchにコピーして利用。
```

### 6. EDINET-BENCH利用外データを使ったLLMの予測
5で配置したデータを使って推論します。  

```
$ python src/edinet_bench/predict.py --task earnings_forecast --model "vertex-ai/claude-3-7-sonnet-20250219" --custom_data_path data/own --custom_data_pattern "**/*.csv" --sheets bs cf pl summary text --wandb false
```

### 7. EDINET-BENCH利用外データ予測に対する業種/売上規模別の精度評価
カスタムデータの結果ファイルを指定して、売上規模別・業種別の精度を出力します。ファイル名は、pyファイル内で指定しているため、適切に変更して利用してください。

```
$ python check_data/src/result_summary.py
```

