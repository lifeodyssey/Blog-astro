---
title: 機械学習における確率的手法
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
  - tensorflow
categories:
  - 学習ノート
abbrlink: 62469dc9
slug: probabilistic-method-in-machine-learning
date: 2022-02-12 14:24:22
mathjax: true
copyright: true
lang: ja
---

機械学習における統計的手法、主にMixture Density Networkの記事から続いています。

<!-- more -->

MDNの記事で主に使用されているのは tfp.distributions.MixtureSameFamily(prob, dist)、

mix.log_prob(y)

MixtureSameFamily、categorical、

```python
def loss(self, y, output):
    prior, mu, scale = self._parse_outputs(output)
    # 出力結果から3つのパラメータを取得
    dist  = getattr(tfp.distributions, self.distribution)(mu, scale)
    prob  = tfp.distributions.Categorical(probs=prior)
    mix   = tfp.distributions.MixtureSameFamily(prob, dist)
    likelihood = mix.log_prob(y)
    return tf.reduce_mean(-likelihood) + tf.add_n([0.] + self.model.losses)
```

これがその関数です。

内部で呼び出されているものを詳しく見ていきましょう。

# 第一行

```python
 dist  = getattr(tfp.distributions, self.distribution)(mu, scale)
```

getattrはオブジェクトの属性値を取得します

```python
getattr(object, name[, default])
```

- object -- オブジェクト
- name -- 文字列、オブジェクト属性
- default -- デフォルトの戻り値。指定しない場合、対応する属性がないとAttributeErrorが発生

ただし、ここではgetattrを使って関数を呼び出す方法を使用しています。

self.distributionは'MultivariateNormalTriL'に等しいです。この行は実際にtfp.distributionsから'MultivariateNormalTriL'関数インスタンスを呼び出し、(mu, scale)をパラメータとして渡しています。

簡単に言えば、分散と平均の2つのパラメータを渡して、多変量正規分布を計算します。

# 第二行

分布を取得した後：

```python
prob  = tfp.distributions.Categorical(probs=prior)
```

カテゴリカル分布は離散分布です。

また数学を復習する必要があります。

## カテゴリカル分布

ベルヌーイ分布は、2つの可能な値しか持たない単一の確率変数の分布です。

多項分布またはカテゴリカル分布は、k個の可能な値を持つ単一の確率変数の分布で、kは有限です。例えば、人間の血液型。

ここでpriorは重みで、確率を重みとしてカテゴリカル分布を作成しています。

# 第三行

```python
mix   = tfp.distributions.MixtureSameFamily(prob, dist)
```

MixtureSameFamilyは同じファミリーの混合分布を実装します。

ここでprobはpdfです。PDFは確率密度関数で、y軸は確率密度であり、確率ではありません。CDFのy軸が確率に対応し、総積分面積は1です。

面積が1であることをほとんど信じられなかったので、手動で計算しました：

```python
from scipy import integrate
v, err = integrate.quad(gm.prob, -2, 3)
print(v)
```

```
0.9999778336488041
```

確かに1です。

# 次の行

```python
likelihood = mix.log_prob(y)
```

また分からなくなりました - なぜ尤度がlog pdfなのでしょうか？

まず尤度関数の定義を見てみましょう。これは与えられた結合サンプル値Xの下での（未知の）パラメータ$\theta$の関数です：
$$
L(\theta|x)=f(x|\theta)
$$

定義上、尤度関数と密度関数は完全に異なる**数学的対象**です：前者は$\theta$の関数、後者は$\textbf{x}$の関数です。

正直言って、完全には理解できませんでした。ここに置いておきましょう。

参考文献：

https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/mixture_same_family.py

https://lulaoshi.info/machine-learning/linear-model/maximum-likelihood-estimation

# 最後の行

```python
tf.reduce_mean(-likelihood) + tf.add_n([0.] + self.model.losses)
```

tf.reduce_meanはlikelihoodの平均を計算します。

tf.add_n([p1, p2, p3....])はリストの要素の加算を実装します。入力オブジェクトはリストで、要素はベクトル、行列などです。

self.model.losses - これは本当によく分かりません。ここ以外で使われているのを見つけられませんでした。

# デバッグを実行して確認

この関数が呼び出されているのを見つけられませんでした。

## MDNコード解析

デフォルトのモデルパラメータは：

以前知らなかったパラメータがいくつかあります：

batchは全サンプルから取得したデータのバッチを指します。

ニューラルネットワークは通常大量のデータを必要とします。すべてのデータを一度にモデルに投入すると、計算量が多すぎてメモリがオーバーフローします。

1サンプルずつ実行すると、訓練速度が遅すぎます。

そのため、通常は適切なサンプルサイズを見つけて、並列計算で訓練を高速化しつつ、一度に処理するデータ量が多すぎないようにします。

imputationsはデータ補完です。

'l2':0.001, 'lr':0.001 - 説明不要です。

残りも同様です。

まだいくつかのコーディングの詳細があり、別の記事で書く必要があります。

自分のコーディング力は本当に不足しています。
