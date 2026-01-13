---
title: Tennary plot
tags:
  - Research
  - python
categories: 学习笔记
mathjax: true
abbrlink: 2b01b12c
date: 2021-03-11 17:19:10
copyright:
---

Description for ternary plot

<!-- more -->

# Python package

This is the package I used https://github.com/marcharper/python-ternary

The following is the most usefull sample for me.

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

# How to read tennary plot

这个才是重头戏

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/ternary_3.jpg)

一个简单的理解是，三元图是重心图，离哪个顶点近，哪个占比大。

由点向底边做两条平行线建立小正三角形，将底边分成三段，中间为顶部组所占比例，左段为右侧组比例，右段为左侧组比例。

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/512.png)

# 我录个视频吧

发现自己在中文网站上找不到相关的信息