---
title: matplotlib-subplot
date: 2020-12-16 20:51:26
tags:
   - Research
   - python
categories: 学習ノート
mathjax: true
abbrlink: bf6c0df1
copyright:
lang: ja
---

今日発見したいくつかのこと

<!-- more -->

# サブプロットの間隔調整

出典：https://blog.csdn.net/qq_33039859/article/details/79424858

```python
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=None)
left  = 0.125  # figureのサブプロットの左側
right = 0.9    # figureのサブプロットの右側
bottom = 0.1   # figureのサブプロットの下側
top = 0.9      # figureのサブプロットの上側
wspace = 0.2   # サブプロット間の空白幅の量、
               # 平均軸幅の割合として表現
hspace = 0.2   # サブプロット間の空白高さの量、
               # 平均軸高さの割合として表現
```

# 軸の共有

`pyplot.subplot()`、`pyplot.axes()`関数、または`Figure.add_subplot()`、`Figure.add_axes()`メソッドで`Axes`を作成する際、`sharex`キーワード引数に別の`Axes`を渡してX軸を共有できます。また、`sharey`キーワード引数に別の`Axes`を渡してY軸を共有できます。**軸を共有すると、一方の`Axes`をズームすると、もう一方の`Axes`も連動してズームします。**

```python
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot([1,2,3,4,5])
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot([7,6,5,4,3,2,1])
```

![1252882-20191225232904746-1316411337](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/1252882-20191225232904746-1316411337.png)
