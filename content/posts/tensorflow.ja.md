---
title: TensorFlow基礎
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
  - tensorflow
categories:
  - 学習ノート
abbrlink: fd169b9d
slug: tensorflow-basics
date: 2022-01-04 10:43:02
mathjax: true
copyright: true
lang: ja
---

ニューラルネットワークの訓練を始めて2日目。

<!-- more -->

# シンプルなニューラルネットワーク構築プロセス

```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

このシンプルなニューラルネットワークの構造を見てみましょう。

`x_train, x_test = x_train / 255.0, x_test / 255.0`はMNISTデータセットが0-255なので、正規化を行っています。

`tf.keras.models.Sequential`は計算のコネクタで、このネットワークが以下のステップで接続され、順番に計算されることを意味します。

最初の層`tf.keras.layers.Flatten`は画像フォーマットを2D配列（28 x 28ピクセル）から1D配列（28 x 28 = 784ピクセル）に変換します。この層は学習するパラメータがなく、データを再フォーマットするだけです。

Dropoutは過学習を防ぐための正則化トリックです。各訓練時にランダムにいくつかのニューロンを無視します。

model.compileには3つのものが必要です：

- *損失関数* - 訓練中のモデル精度を測定。この関数を最小化してモデルを正しい方向に「導く」。
- *オプティマイザ* - データと損失関数に基づいてモデルがどのように更新されるかを決定。
- *メトリクス* - 訓練とテストステップを監視。

# 回帰

ほとんどのプロセスは前と同じです。

# 過学習と過少学習

過学習を防ぐ最も簡単な方法は、小さなモデルから始めることです。学習可能なパラメータが少ないモデル（層の数と各層のユニット数で決まる）。

# ハイパーパラメータ調整

Keras Tunerは、TensorFlowプログラムに最適なハイパーパラメータセットを選択するのに役立つライブラリです。

# ファインチューニング

このチュートリアルでは、転移学習を使用して事前訓練済みネットワークで猫と犬の画像を分類する方法を学びます。

# 自分の問題に戻る

H5フォーマットを使用する必要があるので、まず自分でモデルを再作成する必要があります。

ソースコードを確認する限り、何も問題ではありません。

```python
Fmodel.add(Dense(100, input_shape=(7,),activation='tanh',use_bias=True))
Fmodel.layers[0].set_weights([np.array(f['Weights']['Layer1']).T,np.array(f['Bias']['Layer1']).flatten()])
```

やっと解決しました。あとはファインチューニングをするだけです。

