---
title: '自動機械学習コード詳解'
copyright: true
tags:
  - Automatic Machine Learning
  - Fishery Forecasting
  - Undergraduate Thesis
categories: Papers And Thesis
mathjax: true
abbrlink: 4668cbbb
slug: automl-code-explained
date: 2019-05-16 10:22:38
lang: ja
---

卒業論文ではauto-sklearnに基づいた自動機械学習モデルを応用しました。モデルの原理については[こちら](http://codewithzhangyi.com/2018/07/26/AutoML/)をクリックしてください。ここでは私のコードの使い方を説明します。詳細なコメント版です。

コードリポジトリ：<https://github.com/lifeodyssey/AMF>

<!-- more -->

サーバーのamfディレクトリに移動し、以下を入力します：

```bash
source py3/bin/activate
```

この後、各行の先頭に(py3)と表示され、仮想環境に入ったことを示します。

その後、以下を入力するだけです：

```bash
python code.py
```

プログラムの実行が開始されます。

以下はプログラムの詳細な説明です。コードを修正したい場合は、サーバー上で直接vimを使用するか、ローカルで修正してPSFTPでサーバーにプッシュできます。

```python
import autosklearn.regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 最初の4行は必要なパッケージをインポート
rawdata=pd.read_excel('rawdata.xlsx')  # 引用符内にファイルパスを入力
Y=rawdata[['cpue']]  # 引用符内に予測する変数名を入力

Y=np.log10(Y+1)  # 対数変換を適用、削除可能
X=rawdata[['lon','lat','sst','chla','doy']]  # 使用する変数名を入力
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=7)  # test_sizeはテストセットの割合、0.3は30%
```

```python
automl = autosklearn.regression.AutoSklearnRegressor(
    include_estimators=["random_forest","decision_tree","gradient_boosting","xgradient_boosting"],
    # 効果的と思われるモデルのみを含む
    # モデルの種類はhttps://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regressionを参照
    exclude_estimators=None,
    include_preprocessors=["no_preprocessing", ],
    exclude_preprocessors=None,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 10},  # 10分割交差検証で最適パラメータを決定
    )
automl.fit(x_train, y_train.values.ravel())
```

```python
automl.sprint_statistics()
automl.show_models()
automl.refit(x_train, y_train.values.ravel())
# 新しい独立変数で予測する場合は、以下の行の#を削除
#predata=pd.read_excel('rawdata.xlsx')  # 引用符内にファイルパスを入力
#X=predata[['lon','lat','sst','chla','doy']]  # 引用符内に選択した変数名を入力
y_pre = automl.predict(X)
ypre=np.power(10,ypre)-1  # 変換を戻す
ypre=pd.DataFrame(ypre)
result=pd.concat([rawdata,ypre])  # 新しい独立変数を使用する場合は、rawdataをpredataに変更
result.to_excel('result.xlsx')
```
