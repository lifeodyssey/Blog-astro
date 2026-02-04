---
title: 三元図
tags:
  - Research
  - python
categories: 学習ノート
mathjax: true
abbrlink: 2b01b12c
date: 2021-03-11 17:19:10
copyright:
lang: ja
---

三元図の説明

<!-- more -->

# Pythonパッケージ

使用したパッケージはこちらです https://github.com/marcharper/python-ternary

以下は私にとって最も役立つサンプルです。

```python
import ternary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

cm = plt.cm.get_cmap('viridis')


dat=pd.DataFrame(columns=list('XYZV'))

dat['X']=random.sample(range(45),10)
dat['Y']=random.sample(range(45),10)
dat['Z']=100-(dat['X']+dat['Y'])
dat['V']=10**np.random.randint(0,high=10,size=10)/1e5



# sc = plt.scatter(dat.X, dat.Y, c=dat.V, vmin=0, vmax=7, s=35, cmap=cm)



scale=100



fig, tax = ternary.figure(scale=scale)
fig.set_size_inches(10, 9)

points=dat[['X','Y','Z']].values


minv=np.log10(dat.V.min())
maxv=np.log10(dat.V.max())

tax.scatter(points,s=60,vmin=minv,vmax=maxv,colormap=plt.cm.viridis,colorbar=True,c=np.log10(dat['V'].values),cmap=plt.cm.viridis)


# Decoration.
tax.boundary(linewidth=1)
tax.gridlines(multiple=10, color="gray")
tax.ticks(axis='lbr', linewidth=1, multiple=20)
tax.get_axes().axis('off')
figure.set_size_inches(10, 10)
tax.show()
```

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/D75BADEA-86A3-4AA9-81C8-159829A7F2F7.png)

# 三元図の読み方

これが本題です。

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/ternary_3.jpg)

簡単に理解すると、三元図は重心図です。頂点に近いほど、その成分の割合が大きくなります。

点から底辺に2本の平行線を引いて小さな正三角形を作り、底辺を3つのセグメントに分割します。中央のセグメントは上部成分の割合を表し、左のセグメントは右側成分の割合を表し、右のセグメントは左側成分の割合を表します。

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/512.png)

# 動画を録画しようかな

中国語のウェブサイトで関連情報が見つからなかったので。

