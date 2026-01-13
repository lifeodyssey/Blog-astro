---
title: Keras Tuner& Transfer Learning
tags:
  - 机器学习
  - Inversion
  - Deep Learning
  - tensorflow
categories:
  - 学习笔记
abbrlink: '56104174'
date: 2022-02-24 10:41:27
mathjax:
copyright:
---

看起来这个东西可以让我炼丹更快一点

<!-- more -->

# Keras Tuner

这个东西需要独立安装，python3.6+和tensorflow2.0+

```bash
pip install keras-tuner --upgrade
```

## 一个快速上手的例子

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

接着定义一个tuner

```python
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)
```

接着开始就行

```python
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
```

## 来康康细节

实际上上面那个和我要做的并不太一样

我要做的是调节模型参数，而不是超参数

如果不需要超参数的话，那要做的可能就是重新fit，也就是常说的transfer learning了

#　Transfer Learning

在[上一篇](https://lifeodyssey.github.io/posts/fd169b9d.html)里已经写了一点

然后我重新看了下发现真的非常简单

只要把需要训练的参数固定好，然后重新调用fit就行了



