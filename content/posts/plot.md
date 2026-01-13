---
title: Basis for subplot2grid, ticker, enumerate, seaborn, plt/ax/fig
tags:
  - Research
  - python
categories: 学习笔记
mathjax: true
abbrlink: bebb8f89
copyright: true
date: 2020-08-20 21:34:59
---

一些在读别人代码的时候发现自己还没有掌握的东西。

<!-- more -->

# matplotlib.ticker

## 定位

### Tick locating

The Locator class is the base class for all tick locators. The locators handle autoscaling of the view limits based on the data limits, and the choosing of tick locations. A useful semi-automatic tick locator is [`MultipleLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MultipleLocator). It is initialized with a base, e.g., 10, and it picks axis limits and ticks that are multiples of that base.

The Locator subclasses defined here are

- [`AutoLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.AutoLocator)

  [`MaxNLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MaxNLocator) with simple defaults. This is the default tick locator for most plotting.

- [`MaxNLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MaxNLocator)

  Finds up to a max number of intervals with ticks at nice locations.

- [`LinearLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LinearLocator)

  Space ticks evenly from min to max.

- [`LogLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogLocator)

  Space ticks logarithmically from min to max.

- [`MultipleLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MultipleLocator)

  Ticks and range are a multiple of base; either integer or float.

- [`FixedLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.FixedLocator)

  Tick locations are fixed.

- [`IndexLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.IndexLocator)

  Locator for index plots (e.g., where `x = range(len(y))`).

- [`NullLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.NullLocator)

  No ticks.

- [`SymmetricalLogLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.SymmetricalLogLocator)

  Locator for use with with the symlog norm; works like [`LogLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogLocator) for the part outside of the threshold and adds 0 if inside the limits.

- [`LogitLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogitLocator)

  Locator for logit scaling.

- [`OldAutoLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.OldAutoLocator)

  Choose a [`MultipleLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.MultipleLocator) and dynamically reassign it for intelligent ticking during navigation.

- [`AutoMinorLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.AutoMinorLocator)

  Locator for minor ticks when the axis is linear and the major ticks are uniformly spaced. Subdivides the major tick interval into a specified number of minor intervals, defaulting to 4 or 5 depending on the major interval.

There are a number of locators specialized for date locations - see the `dates` module.

You can define your own locator by deriving from Locator. You must override the `__call__` method, which returns a sequence of locations, and you will probably want to override the autoscale method to set the view limits from the data limits.

If you want to override the default locator, use one of the above or a custom locator and pass it to the x or y axis instance. The relevant methods are:

```python
ax.xaxis.set_major_locator(xmajor_locator)
ax.xaxis.set_minor_locator(xminor_locator)
ax.yaxis.set_major_locator(ymajor_locator)
ax.yaxis.set_minor_locator(yminor_locator)
```



The default minor locator is [`NullLocator`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.NullLocator), i.e., no minor ticks on by default.

## 格式

### Tick formatting

Tick formatting is controlled by classes derived from Formatter. The formatter operates on a single tick value and returns a string to the axis.

- [`NullFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.NullFormatter)

  No labels on the ticks.

- [`IndexFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.IndexFormatter)

  Set the strings from a list of labels.

- [`FixedFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.FixedFormatter)

  Set the strings manually for the labels.

- [`FuncFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.FuncFormatter)

  User defined function sets the labels.

- [`StrMethodFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.StrMethodFormatter)

  Use string [`format`](https://docs.python.org/3/library/functions.html#format) method.

- [`FormatStrFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.FormatStrFormatter)

  Use an old-style sprintf format string.

- [`ScalarFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.ScalarFormatter)

  Default formatter for scalars: autopick the format string.

- [`LogFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogFormatter)

  Formatter for log axes.

- [`LogFormatterExponent`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogFormatterExponent)

  Format values for log axis using `exponent = log_base(value)`.

- [`LogFormatterMathtext`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogFormatterMathtext)

  Format values for log axis using `exponent = log_base(value)` using Math text.

- [`LogFormatterSciNotation`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogFormatterSciNotation)

  Format values for log axis using scientific notation.

- [`LogitFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.LogitFormatter)

  Probability formatter.

- [`EngFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.EngFormatter)

  Format labels in engineering notation

- [`PercentFormatter`](https://matplotlib.org/3.1.1/api/ticker_api.html#matplotlib.ticker.PercentFormatter)

  Format labels as a percentage

You can derive your own formatter from the Formatter base class by simply overriding the `__call__` method. The formatter class has access to the axis view and data limits.

To control the major and minor tick label formats, use one of the following methods:

```python
ax.xaxis.set_major_formatter(xmajor_formatter)
ax.xaxis.set_minor_formatter(xminor_formatter)
ax.yaxis.set_major_formatter(ymajor_formatter)
ax.yaxis.set_minor_formatter(yminor_formatter)
```



See [Major and minor ticks](https://matplotlib.org/3.1.1/gallery/ticks_and_spines/major_minor_demo.html) for an example of setting major and minor ticks. See the [`matplotlib.dates`](https://matplotlib.org/3.1.1/api/dates_api.html#module-matplotlib.dates) module for more information and examples of using date locators and formatters.

参考

https://matplotlib.org/3.1.1/api/ticker_api.html

https://matplotlib.org/3.1.1/api/ticker_api.html

# enumerate

## 描述

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

Python 2.3. 以上版本可用，2.6 添加 start 参数。

### 语法

以下是 enumerate() 方法的语法:

```python
enumerate(sequence, [start=0])
```

### 参数

- sequence -- 一个序列、迭代器或其他支持迭代对象。
- start -- 下标起始位置。

### 返回值

返回 enumerate(枚举) 对象。

------

## 实例

以下展示了使用 enumerate() 方法的实例：

```python
>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```



### 普通的 for 循环

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

### for 循环使用 enumerate

```python
>>>seq = ['one', 'two', 'three']
>>> for i, element in enumerate(seq):
...     print i, element
... 
0 one参考
1 two
2 three
```

参考

https://www.runoob.com/python/python-func-enumerate.html

# subplot2grid

这个可以自定义子图的位置，并且可以跨越原来大小。

原文https://wizardforcel.gitbooks.io/matplotlib-user-guide/content/3.3.html

```python
GridSpec
```

指定子图将放置的网格的几何位置。 需要设置网格的行数和列数。 子图布局参数（例如，左，右等）可以选择性调整。

```python
SubplotSpec
```

指定在给定`GridSpec`中的子图位置。

```python
subplot2grid
```

一个辅助函数，类似于`pyplot.subplot`，但是使用基于 0 的索引，并可使子图跨越多个格子。

## subplot2grid基本示例

要使用subplot2grid，你需要提供网格的几何形状和网格中子图的位置。 对于简单的单网格子图：

```python
ax = plt.subplot2grid((2,2),(0, 0))
```

等价于：

```python
ax = plt.subplot(2,2,1)
       nRow=2, nCol=2
(0,0) +-------+-------+
      |   1   |       |
      +-------+-------+
      |       |       |
      +-------+-------+
```

要注意不像`subplot`，`gridspec`中的下标从 0 开始。

为了创建跨越多个格子的子图，

```python
ax2 = plt.subplot2grid((3,3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
```

例如，下列命令：

```python
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3,3), (2, 0))
ax5 = plt.subplot2grid((3,3), (2, 1))
```

会创建：

![img](http://matplotlib.org/_images/demo_gridspec01.png)

## GridSpec和SubplotSpec

你可以显式创建`GridSpec`并用它们创建子图。

例如，

```python
ax = plt.subplot2grid((2,2),(0, 0))
```

等价于：

```python
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 2)
ax = plt.subplot(gs[0, 0])
```

`gridspec`示例提供类似数组（一维或二维）的索引，并返回`SubplotSpec`实例。例如，使用切片来返回跨越多个格子的`SubplotSpec`实例。

上面的例子会变成：

```python
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1,:-1])
ax3 = plt.subplot(gs[1:, -1])
ax4 = plt.subplot(gs[-1,0])
ax5 = plt.subplot(gs[-1,-2])
```

![img](http://matplotlib.org/_images/demo_gridspec02.png)

## 调整 GridSpec布局

在显式使用`GridSpec`的时候，你可以调整子图的布局参数，子图由`gridspec`创建。

```python
gs1 = gridspec.GridSpec(3, 3)
gs1.update(left=0.05, right=0.48, wspace=0.05)
```

这类似于`subplots_adjust`，但是他只影响从给定`GridSpec`创建的子图。

下面的代码

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

会产生

![img](http://matplotlib.org/_images/demo_gridspec03.png)

## 使用 SubplotSpec创建 GridSpec

你可以从`SubplotSpec`创建`GridSpec`，其中它的布局参数设置为给定`SubplotSpec`的位置的布局参数。

```python
gs0 = gridspec.GridSpec(1, 2)

gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
gs01 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[1])
```

![img](http://matplotlib.org/_images/demo_gridspec04.png)

## 使用SubplotSpec创建复杂嵌套的GridSpec

这里有一个更复杂的嵌套`gridspec`的示例，我们通过在每个 3x3 内部网格中隐藏适当的脊线，在 4x4 外部网格的每个单元格周围放置一个框。

![img](http://matplotlib.org/_images/demo_gridspec06.png)

## 网格尺寸可变的GridSpec

通常，`GridSpec`创建大小相等的网格。你可以调整行和列的相对高度和宽度，要注意绝对高度值是无意义的，有意义的只是它们的相对比值。

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

seaborn是对matplotlib进一步的封装，简单点来说就是更简单了。

官网https://seaborn.pydata.org/

我这里放几个我感觉用得上的代码。

## lineplot


seaborn.lineplot(x=None, y=None, hue=None, size=None, style=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, dashes=True, markers=None, style_order=None, units=None, estimator='mean', ci=95, n_boot=1000, seed=None, sort=True, err_style='band', err_kws=None, legend='brief', ax=None, **kwargs)

https://seaborn.pydata.org/generated/seaborn.lineplot.html

## heatmap

seaborn.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)

https://zhuanlan.zhihu.com/p/35494575

## implot

eaborn.lmplot(*x*, *y*, *data*, *hue=None*, *col=None*, *row=None*, *palette=None*, *col_wrap=None*, *size=5*, *aspect=1*, *markers='o'*, *sharex=True*, *sharey=True*, *hue_order=None*, *col_order=None*, *row_order=None*, *legend=True*, *legend_out=True*, *x_estimator=None*, *x_bins=None*, *x_ci='ci'*, *scatter=True*, *fit_reg=True*, *ci=95*, *n_boot=1000*, *units=None*, *order=1*, *logistic=False*, *lowess=False*, *robust=False*, *logx=False*, *x_partial=None*, *y_partial=None*, *truncate=False*, *x_jitter=None*, *y_jitter=None*, *scatter_kws=None*, *line_kws=None*)

https://zhuanlan.zhihu.com/p/25909753

常见统计图片基本都可以在这里面看到

https://seaborn.pydata.org/examples/index.html

## subplot_adjust

```python
matplotlib.pyplot.subplots_adjust(*args, **kwargs)
subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)

left  = 0.125  # 子图(subplot)距画板(figure)左边的距离
right = 0.9    # 右边
bottom = 0.1   # 底部
top = 0.9      # 顶部
wspace = 0.2   # 子图水平间距
hspace = 0.2   # 子图垂直间距
```

# plt/ax/fig

![img](https://pic2.zhimg.com/80/v2-6e4429872eeb8a155433c0ee7c75b6ea_720w.jpg)

尽量避免直接使用plt

https://zhuanlan.zhihu.com/p/93423829