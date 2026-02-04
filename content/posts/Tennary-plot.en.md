---
title: Ternary plot
tags:
  - Research
  - python
categories: Learning Notes
mathjax: true
abbrlink: 2b01b12c
date: 2021-03-11 17:19:10
copyright:
lang: en
---

Description for ternary plot

<!-- more -->

# Python package

This is the package I used https://github.com/marcharper/python-ternary

The following is the most useful sample for me.

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

# How to read ternary plot

This is the main topic.

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/ternary_3.jpg)

A simple understanding is that a ternary plot is a centroid diagram - the closer to a vertex, the larger the proportion of that component.

Draw two parallel lines from a point to the base to create a small equilateral triangle, dividing the base into three segments. The middle segment represents the proportion of the top component, the left segment represents the proportion of the right component, and the right segment represents the proportion of the left component.

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/512.png)

# Maybe I should record a video

I found that I couldn't find relevant information on Chinese websites.

