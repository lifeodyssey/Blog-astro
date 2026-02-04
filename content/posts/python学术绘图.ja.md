---
title: Python描画基礎
tags:
  - Research
  - python
categories: 学習ノート
mathjax: true
abbrlink: '720e5580'
copyright: true
date: 2020-08-21 16:31:41
lang: ja
---

この2日間、描画の整理をしていて、matplotlibの知識が本当に不足していることに気づきました。このノートはcartopy以外で使うものを整理するためのものです。

<!-- more -->

# matplotlib

## 基礎知識

Matplotlibで最も重要な基礎概念はfigureとaxesです。

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/v2-6e4429872eeb8a155433c0ee7c75b6ea_720w.jpg)

Matplotlibでは、figureは描画ボード、axesはキャンバス、axisは座標軸を意味します。例えば：

```python
x = np.linspace(0, 10, 20)  # データ生成
y = x * x + 2

fig = plt.figure()  # 新しいfigureオブジェクトを作成
axes = fig.add_axes([0.5, 0.5, 0.8, 0.8])  # キャンバスの左、下、幅、高さを制御
axes.plot(x, y, 'r')
```

同じ描画ボード上に複数のキャンバスを描くことができます：

```python
fig = plt.figure()  # 新しい描画ボードを作成
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # 大きいキャンバス
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])  # 小さいキャンバス

axes1.plot(x, y, 'r')  # 大きいキャンバス
axes2.plot(y, x, 'g')  # 小さいキャンバス
```

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/output_80_1.png)

また、plt.subplots()を使ってキャンバスを追加する方法もあります：

```python
fig, axes = plt.subplots(nrows=1, ncols=2)  # サブプロット：1行、2列
for ax in axes:
    ax.plot(x, y, 'r')
```

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/output_86_0.png)

1つのキャンバスだけを描く場合でも、plt.plot()ではなくfig, axes = plt.subplots()を使用することをお勧めします。

## 基本スタイル調整

### タイトルと凡例の追加

タイトル、軸ラベル、凡例を含む図形を描画：

```python
fig, axes = plt.subplots()

axes.set_xlabel('x label')  # X軸ラベル
axes.set_ylabel('y label')
axes.set_title('title')  # 図のタイトル

axes.plot(x, x**2)
axes.plot(x, x**3)
axes.legend(["y = x**2", "y = x**3"], loc=0)  # 凡例
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_98_1.png)

凡例の`loc`パラメータは凡例の位置を示します。`1, 2, 3, 4`はそれぞれ右上、左上、左下、右下を表し、`0`は自動適応を意味します。

### 線のスタイル、色、透明度

```python
fig, axes = plt.subplots()

axes.plot(x, x+1, color="red", alpha=0.5)
axes.plot(x, x+2, color="#1155dd")
axes.plot(x, x+3, color="#15cc55")
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_103_1.png)

### グリッドと軸範囲

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# グリッドを表示
axes[0].plot(x, x**2, x, x**3, lw=2)
axes[0].grid(True)

# 軸範囲を設定
axes[1].plot(x, x**2, x, x**3)
axes[1].set_ylim([0, 60])
axes[1].set_xlim([2, 5])
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_110_1.png)

## チートシート

Matplotlib公式がチートシートを提供しています：

https://github.com/matplotlib/cheatsheets

## 3Dグラフィックス

Matplotlibは3Dグラフィックスも描画できます。3Dグラフィックスは主に`mplot3d`モジュールで実装されます。

```python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_14_1.png)

3D曲面プロット：

```python
fig = plt.figure()
ax = Axes3D(fig)

X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X ** 2 + Y ** 2)

ax.plot_surface(X, Y, Z, cmap=plt.cm.winter)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_25_1_1.png)

# seaborn

SeabornはMatplotlibコアライブラリの上に構築された高レベルAPIで、より美しいグラフィックスを簡単に描画できます。

## クイック最適化

```python
import seaborn as sns

sns.set()  # Seabornスタイルを宣言

plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_15_1.png)

## 関連プロット

```python
iris = sns.load_dataset("iris")
sns.relplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_36_1.png)

## カテゴリプロット

カテゴリプロットのFigureレベルインターフェースは`catplot`です。以下が含まれます：

- カテゴリ散布図：`stripplot()`、`swarmplot()`
- カテゴリ分布：`boxplot()`、`violinplot()`、`boxenplot()`
- カテゴリ推定：`pointplot()`、`barplot()`、`countplot()`

```python
sns.catplot(x="sepal_length", y="species", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_54_1.png)

箱ひげ図：

```python
sns.catplot(x="sepal_length", y="species", kind="box", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_59_1.png)

バイオリンプロット：

```python
sns.catplot(x="sepal_length", y="species", kind="violin", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_62_2.png)

## 分布プロット

分布プロットは変数の分布を可視化するために使用されます。Seabornは`jointplot`、`pairplot`、`distplot`、`kdeplot`を提供しています。

```python
sns.distplot(iris["sepal_length"])
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_80_2.png)

`jointplot`は二変量分布に使用されます：

```python
sns.jointplot(x="sepal_length", y="sepal_width", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_86_2.png)

`pairplot`はすべての特徴量のペアワイズ比較をサポートします：

```python
sns.pairplot(iris, hue="species")
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_101_2.png)

## 回帰プロット

回帰プロット関数：`lmplot`と`regplot`。

```python
sns.regplot(x="sepal_length", y="sepal_width", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_106_2.png)

`lmplot`は比較のために第三次元を導入することをサポートします：

```python
sns.lmplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_109_2.png)

## マトリックスプロット

最もよく使われるのは`heatmap`と`clustermap`です。

```python
import numpy as np

sns.heatmap(np.random.rand(10, 10))
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_114_1.png)

ヒートマップは特定のシナリオで非常に便利です。例えば、変数の相関係数ヒートマップを描画する場合などです。

