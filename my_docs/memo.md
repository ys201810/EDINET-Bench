# 20255/7/12
## random-forestを作る。
デフォルトだとAUCが0.52で下がった。
--n_estimators 200 --max_depth 10で同等。

python src/edinet_bench/random_forest.py --task earnings_forecast --n_estimators 200 --max_depth 10で実行。
logistic回帰
```
Accuracy: 0.568, Precision: 0.669, Recall: 0.667, F1: 0.668
[[ 60  97]
 [ 98 196]]
AUC: 0.561
```

ランダムフォレスト
```
Accuracy: 0.601, Precision: 0.689, Recall: 0.707, F1: 0.698
[[ 63  94]
 [ 86 208]]
AUC: 0.555
```

## gbdtを作る。
gbdt
python src/edinet_bench/gbdt.py --task earnings_forecast
```
Accuracy: 0.585, Precision: 0.689, Recall: 0.663, F1: 0.676
[[ 69  88]
 [ 99 195]]
AUC: 0.561
```

xgboost
python src/edinet_bench/xgboost_model.py --task earnings_forecast

```
Accuracy: 0.570, Precision: 0.695, Recall: 0.605, F1: 0.647
[[ 79  78]
 [116 178]]
AUC: 0.563
```

lightGBM
python src/edinet_bench/lightgbm_model.py --task earnings_forecast
```
Accuracy: 0.583, Precision: 0.693, Recall: 0.646, F1: 0.669
[[ 73  84]
 [104 190]]
AUC: 0.579
```

## 差分特徴量を作る。
logistic回帰 + 差分特徴量
python src/edinet_bench/logistic.py --task earnings_forecast --use_differential_features

```
Accuracy: 0.539, Precision: 0.672, Recall: 0.571, F1: 0.618
[[ 75  82]
 [126 168]]
AUC: 0.551
```

random_forest + 差分特徴量
python src/edinet_bench/random_forest.py --task earnings_forecast --use_differential_features --n_estimators 200 --max_depth 10

```
Accuracy: 0.579, Precision: 0.687, Recall: 0.650, F1: 0.668
[[ 70  87]
 [103 191]]
AUC: 0.593
```

gbdt + 差分特徴量
python src/edinet_bench/gbdt.py --task earnings_forecast --use_differential_features
```
Accuracy: 0.565, Precision: 0.708, Recall: 0.568, F1: 0.630
[[ 88  69]
 [127 167]]
AUC: 0.561
```

xgboost + 差分特徴量
python src/edinet_bench/xgboost_model.py --task earnings_forecast --use_differential_features --n_estimators 200 --max_depth 8
```
Accuracy: 0.585, Precision: 0.720, Recall: 0.595, F1: 0.652
[[ 89  68]
 [119 175]]
AUC: 0.578
```

lightgbm + 差分特徴量
python src/edinet_bench/lightgbm_model.py --task earnings_forecast --use_differential_features --num_leaves 50
```
Accuracy: 0.557, Precision: 0.691, Recall: 0.578, F1: 0.630
[[ 81  76]
 [124 170]]
AUC: 0.585
```

## 割合特徴量を作る。
logistic回帰 + 差分特徴量 + 割合特徴量
Accuracy: 0.545, Precision: 0.677, Recall: 0.578, F1: 0.624
[[ 76  81]
 [124 170]]

random_forest + 差分特徴量 + 割合特徴量
python src/edinet_bench/random_forest.py --task earnings_forecast --use_differential_features  --use_percentage_change_features
Accuracy: 0.563, Precision: 0.693, Recall: 0.592, F1: 0.639
[[ 80  77]
 [120 174]]
AUC: 0.563

gbdt + 差分特徴量 + 割合特徴量
python src/edinet_bench/gbdt.py --task earnings_forecast --use_differential_features  --use_percentage_change_features
Accuracy: 0.557, Precision: 0.703, Recall: 0.554, F1: 0.620
[[ 88  69]
 [131 163]]
AUC: 0.564

xgboost + 差分特徴量 + 割合特徴量

lightgbm + 差分特徴量 + 割合特徴量



### メモ
python src/edinet_bench/random_forest.py --task earnings_forecast --use_differential_features --n_estimators 200 --max_depth 10
が

```
try:
    from .utils import create_differential_features
except ImportError:
    from utils import create_differential_features
```

を
```
from edinet_behcn.utils import create_differential_features
```

に変えると精度が落ちる。なぜか？