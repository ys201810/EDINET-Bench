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

