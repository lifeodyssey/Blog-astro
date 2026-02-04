---
title: subplot2grid、ticker、enumerate、seaborn、plt/ax/figの基礎
tags:
  - Research
  - python
categories: 学習ノート
mathjax: true
abbrlink: bebb8f89
copyright: true
date: 2020-08-20 21:34:59
lang: ja
---

他の人のコードを読んでいる時に、まだ習得していなかったことを発見しました。

<!-- more -->

# matplotlib.ticker

## 位置決め

### Tick locating

Locatorクラスはすべてのティックロケーターの基底クラスです。ロケーターはデータの範囲に基づいてビューの範囲の自動スケーリングとティックの位置の選択を処理します。便利な半自動ティックロケーターは[`MultipleLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MultipleLocator)です。基数（例：10）で初期化され、その基数の倍数である軸の範囲とティックを選択します。

ここで定義されているLocatorサブクラスは以下の通りです

- [`AutoLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.AutoLocator)

  シンプルなデフォルトを持つ[`MaxNLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MaxNLocator)。ほとんどのプロットのデフォルトティックロケーターです。

- [`MaxNLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MaxNLocator)

  適切な位置にティックを持つ最大数の間隔を見つけます。

- [`LinearLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LinearLocator)

  最小から最大まで均等にティックを配置します。

- [`LogLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogLocator)

  最小から最大まで対数的にティックを配置します。

- [`MultipleLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MultipleLocator)

  ティックと範囲は基数の倍数です。整数または浮動小数点数。

- [`FixedLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.FixedLocator)

  ティックの位置は固定されています。

- [`IndexLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.IndexLocator)

  インデックスプロット用のロケーター（例：`x = range(len(y))`の場合）。

- [`NullLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.NullLocator)

  ティックなし。

- [`SymmetricalLogLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.SymmetricalLogLocator)

  symlogノルムで使用するロケーター。閾値外の部分は[`LogLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogLocator)のように動作し、範囲内の場合は0を追加します。

- [`LogitLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogitLocator)

  ロジットスケーリング用のロケーター。

- [`OldAutoLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.OldAutoLocator)

  [`MultipleLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MultipleLocator)を選択し、ナビゲーション中にインテリジェントなティッキングのために動的に再割り当てします。

- [`AutoMinorLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.AutoMinorLocator)

  軸が線形で主ティックが均等に配置されている場合の副ティック用ロケーター。主ティック間隔を指定された数の副間隔に分割し、主間隔に応じてデフォルトで4または5になります。

日付の位置に特化したロケーターがいくつかあります - `dates`モジュールを参照してください。

Locatorから派生して独自のロケーターを定義できます。位置のシーケンスを返す`__call__`メソッドをオーバーライドする必要があり、データの範囲からビューの範囲を設定するautoscaleメソッドもオーバーライドしたいでしょう。

デフォルトのロケーターをオーバーライドする場合は、上記のいずれかまたはカスタムロケーターを使用し、xまたはy軸インスタンスに渡します。関連するメソッドは以下の通りです：

```python
ax.xaxis.set_major_locator(xmajor_locator)
ax.xaxis.set_minor_locator(xminor_locator)
ax.yaxis.set_major_locator(ymajor_locator)
ax.yaxis.set_minor_locator(yminor_locator)
```

デフォルトの副ロケーターは[`NullLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.NullLocator)です。つまり、デフォルトでは副ティックはありません。

## フォーマット

### Tick formatting

ティックのフォーマットはFormatterから派生したクラスによって制御されます。フォーマッターは単一のティック値を操作し、軸に文字列を返します。

- [`NullFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.NullFormatter)

  ティックにラベルなし。

- [`IndexFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.IndexFormatter)

  ラベルのリストから文字列を設定します。

- [`FixedFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.FixedFormatter)

  ラベルの文字列を手動で設定します。

- [`FuncFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.FuncFormatter)

  ユーザー定義関数がラベルを設定します。

- [`StrMethodFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.StrMethodFormatter)

  文字列[`format`](https://docs.python.org/3/library/functions.html#format)メソッドを使用します。

- [`FormatStrFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.FormatStrFormatter)

  古いスタイルのsprintf形式文字列を使用します。

- [`ScalarFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.ScalarFormatter)

  スカラーのデフォルトフォーマッター：形式文字列を自動選択します。

- [`LogFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogFormatter)

  対数軸用のフォーマッター。

- [`LogFormatterExponent`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogFormatterExponent)

  `exponent = log_base(value)`を使用して対数軸の値をフォーマットします。

- [`LogFormatterMathtext`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogFormatterMathtext)

  Math textを使用して`exponent = log_base(value)`で対数軸の値をフォーマットします。

- [`LogFormatterSciNotation`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogFormatterSciNotation)

  科学表記法を使用して対数軸の値をフォーマットします。

- [`LogitFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogitFormatter)

  確率フォーマッター。

- [`EngFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.EngFormatter)

  工学表記法でラベルをフォーマットします。

- [`PercentFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.PercentFormatter)

  ラベルをパーセンテージとしてフォーマットします。

`__call__`メソッドをオーバーライドするだけで、Formatter基底クラスから独自のフォーマッターを派生できます。フォーマッタークラスは軸のビューとデータの範囲にアクセスできます。

主ティックと副ティックのラベル形式を制御するには、以下のメソッドのいずれかを使用します：

```python
ax.xaxis.set_major_formatter(xmajor_formatter)
ax.xaxis.set_minor_formatter(xminor_formatter)
ax.yaxis.set_major_formatter(ymajor_formatter)
ax.yaxis.set_minor_formatter(yminor_formatter)
```

参考

https://matplotlib.org/3.1.1/api/ticker_api.html

# enumerate

## 説明

enumerate()関数は、イテラブルなデータオブジェクト（リスト、タプル、文字列など）をインデックスシーケンスに結合し、データとそのインデックスの両方をリストするために使用されます。一般的にforループで使用されます。

Python 2.3以上で使用可能で、2.6でstartパラメータが追加されました。

### 構文

enumerate()メソッドの構文は以下の通りです：

```python
enumerate(sequence, [start=0])
```

### パラメータ

- sequence -- シーケンス、イテレータ、またはイテレーションをサポートする他のオブジェクト。
- start -- インデックスの開始位置。

### 戻り値

enumerateオブジェクトを返します。

------

## 例

以下はenumerate()メソッドの使用例です：

```python
>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # インデックスは1から開始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

### 通常のforループ

```python
>>>i = 0
>>> seq = ['one', 'two', 'three']
>>> for element in seq:
...     print i, seq[i]
...     i +=1
...
0 one
1 two
2 three
```

### enumerateを使用したforループ

```python
>>>seq = ['one', 'two', 'three']
>>> for i, element in enumerate(seq):
...     print i, element
...
0 one
1 two
2 three
```

参考

https://www.runoob.com/python/python-func-enumerate.html

# subplot2grid

これはサブプロットの位置をカスタマイズし、元のサイズを跨ぐことができます。

原文：https://wizardforcel.gitbooks.io/matplotlib-user-guide/content/3.3.html

```python
GridSpec
```

サブプロットが配置されるグリッドのジオメトリを指定します。グリッドの行数と列数を設定する必要があります。サブプロットのレイアウトパラメータ（例：left、rightなど）はオプションで調整できます。

```python
SubplotSpec
```

指定された`GridSpec`内のサブプロットの位置を指定します。

```python
subplot2grid
```

`pyplot.subplot`に似たヘルパー関数ですが、0ベースのインデックスを使用し、サブプロットが複数のセルにまたがることができます。

## subplot2grid基本例

subplot2gridを使用するには、グリッドのジオメトリとグリッド内のサブプロットの位置を提供する必要があります。単純な単一セルのサブプロットの場合：

```python
ax = plt.subplot2grid((2,2),(0, 0))
```

これは以下と同等です：

```python
ax = plt.subplot(2,2,1)
       nRow=2, nCol=2
(0,0) +-------+-------+
      |   1   |       |
      +-------+-------+
      |       |       |
      +-------+-------+
```

`subplot`とは異なり、`gridspec`のインデックスは0から始まることに注意してください。

複数のセルにまたがるサブプロットを作成するには、

```python
ax2 = plt.subplot2grid((3,3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
```

例えば、以下のコマンド：

```python
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3,3), (2, 0))
ax5 = plt.subplot2grid((3,3), (2, 1))
```

は以下を作成します：

![img](http://matplotlib.org/_images/demo_gridspec01.png)

## GridSpecとSubplotSpec

`GridSpec`を明示的に作成し、それを使用してサブプロットを作成できます。

例えば、

```python
ax = plt.subplot2grid((2,2),(0, 0))
```

は以下と同等です：

```python
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 2)
ax = plt.subplot(gs[0, 0])
```

`gridspec`の例は配列のような（1Dまたは2D）インデックスを提供し、`SubplotSpec`インスタンスを返します。例えば、スライスを使用して複数のセルにまたがる`SubplotSpec`インスタンスを返します。

上記の例は以下のようになります：

```python
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1,:-1])
ax3 = plt.subplot(gs[1:, -1])
ax4 = plt.subplot(gs[-1,0])
ax5 = plt.subplot(gs[-1,-2])
```

![img](http://matplotlib.org/_images/demo_gridspec02.png)

## GridSpecレイアウトの調整

`GridSpec`を明示的に使用する場合、`gridspec`によって作成されたサブプロットのレイアウトパラメータを調整できます。

```python
gs1 = gridspec.GridSpec(3, 3)
gs1.update(left=0.05, right=0.48, wspace=0.05)
```

これは`subplots_adjust`に似ていますが、指定された`GridSpec`から作成されたサブプロットにのみ影響します。

以下のコード

```python
gs1 = gridspec.GridSpec(3, 3)
gs1.update(left=0.05, right=0.48, wspace=0.05)
ax1 = plt.subplot(gs1[:-1, :])
ax2 = plt.subplot(gs1[-1, :-1])
ax3 = plt.subplot(gs1[-1, -1])

gs2 = gridspec.GridSpec(3, 3)
gs2.update(left=0.55, right=0.98, hspace=0.05)
ax4 = plt.subplot(gs2[:, :-1])
ax5 = plt.subplot(gs2[:-1, -1])
ax6 = plt.subplot(gs2[-1, -1])
```

は以下を生成します

![img](http://matplotlib.org/_images/demo_gridspec03.png)

## SubplotSpecからGridSpecを作成

`SubplotSpec`から`GridSpec`を作成できます。そのレイアウトパラメータは指定された`SubplotSpec`の位置のレイアウトパラメータに設定されます。

```python
gs0 = gridspec.GridSpec(1, 2)

gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
gs01 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[1])
```

![img](http://matplotlib.org/_images/demo_gridspec04.png)

## SubplotSpecを使用した複雑なネストされたGridSpecの作成

ここでは、より複雑なネストされた`gridspec`の例を示します。4x4の外側グリッドの各セルの周りに、各3x3の内側グリッドで適切なスパインを非表示にしてボックスを配置します。

![img](http://matplotlib.org/_images/demo_gridspec06.png)

## 可変グリッドサイズのGridSpec

通常、`GridSpec`は同じサイズのグリッドを作成します。行と列の相対的な高さと幅を調整できます。絶対的な高さの値は意味がなく、相対的な比率のみが重要であることに注意してください。

```python
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1,2],
                       height_ratios=[4,1]
                       )

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])
```

![img](http://matplotlib.org/_images/demo_gridspec05.png)

# seaborn

seabornはmatplotlibをさらにカプセル化したもので、簡単に言えばより使いやすくなっています。

公式サイト：https://seaborn.pydata.org/

ここでは使えそうなコードをいくつか紹介します。

## lineplot

seaborn.lineplot(x=None, y=None, hue=None, size=None, style=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, dashes=True, markers=None, style_order=None, units=None, estimator='mean', ci=95, n_boot=1000, seed=None, sort=True, err_style='band', err_kws=None, legend='brief', ax=None, **kwargs)

https://seaborn.pydata.org/generated/seaborn.lineplot.html

## heatmap

seaborn.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)

https://zhuanlan.zhihu.com/p/35494575

## lmplot

seaborn.lmplot(*x*, *y*, *data*, *hue=None*, *col=None*, *row=None*, *palette=None*, *col_wrap=None*, *size=5*, *aspect=1*, *markers='o'*, *sharex=True*, *sharey=True*, *hue_order=None*, *col_order=None*, *row_order=None*, *legend=True*, *legend_out=True*, *x_estimator=None*, *x_bins=None*, *x_ci='ci'*, *scatter=True*, *fit_reg=True*, *ci=95*, *n_boot=1000*, *units=None*, *order=1*, *logistic=False*, *lowess=False*, *robust=False*, *logx=False*, *x_partial=None*, *y_partial=None*, *truncate=False*, *x_jitter=None*, *y_jitter=None*, *scatter_kws=None*, *line_kws=None*)

https://zhuanlan.zhihu.com/p/25909753

一般的な統計グラフは基本的にここで見つけることができます

https://seaborn.pydata.org/examples/index.html

## subplot_adjust

```python
matplotlib.pyplot.subplots_adjust(*args, **kwargs)
subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)

left  = 0.125  # サブプロットからfigure左端までの距離
right = 0.9    # 右端
bottom = 0.1   # 下端
top = 0.9      # 上端
wspace = 0.2   # サブプロット間の水平間隔
hspace = 0.2   # サブプロット間の垂直間隔
```

# plt/ax/fig

![img](https://pic2.zhimg.com/80/v2-6e4429872eeb8a155433c0ee7c75b6ea_720w.jpg)

pltを直接使用することは避けてください

https://zhuanlan.zhihu.com/p/93423829

