---
title: Keras Tuner & Transfer Learning
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
  - tensorflow
categories:
  - 学習ノート
abbrlink: '56104174'
slug: keras-tuner-transfer-learning
date: 2022-02-24 10:41:27
lang: ja
mathjax:
copyright:
---

このツールを使えば、モデルの学習がもっと速くなりそうです。

<!-- more -->

# Keras Tuner

これは別途インストールが必要で、Python 3.6+とTensorFlow 2.0+が必要です。

```bash
pip install keras-tuner --upgrade
```

## クイックスタートの例

```python
from tensorflow import keras
from tensorflow.keras import layers


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
    )
    return model
```

次にtunerを定義します

```python
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)
```

あとは開始するだけです

```python
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
```

## 詳細を見てみましょう

実際、上記の例は私がやりたいこととは少し違います。

私がやりたいのはモデルパラメータの調整であり、ハイパーパラメータではありません。

ハイパーパラメータが不要な場合、やるべきことはおそらく再fitすること、つまり一般的に言われるtransfer learningです。

# Transfer Learning

[前回の記事](https://lifeodyssey.github.io/posts/fd169b9d.html)で少し書きました。

改めて見直してみると、本当にとてもシンプルでした。

学習させたいパラメータを固定して、再度fitを呼び出すだけです。
