---
title: python绘图基础
tags:
  - Research
  - python
categories: 学习笔记
mathjax: true
abbrlink: '720e5580'
copyright: true
date: 2020-08-21 16:31:41
---

这两天在收拾画图的事情，发现我matplotlib这些东西学的真的差，这个笔记用来整理一下自己会用到的东西，除了cartopy之外的。

<!-- more -->

# matplotlib

## 基础知识

Matplotlib最重要一个基础概念就是figure和axes。

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/v2-6e4429872eeb8a155433c0ee7c75b6ea_720w.jpg)

在Matplotlib中，figure的意思是画板，axes的意思是画布，而axis的意思是坐标轴。比如

```python
x = np.linspace(0, 10, 20)  # 生成数据
y = x * x + 2

fig = plt.figure()  # 新建图形对象
axes = fig.add_axes([0.5, 0.5, 0.8, 0.8])  # 控制画布的左，下，宽度，高度
axes.plot(x, y, 'r')
```

在同一个画板上，我们可以画好几个画布

```python
fig = plt.figure()  # 新建画板
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # 大画布
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])  # 小画布

axes1.plot(x, y, 'r')  # 大画布
axes2.plot(y, x, 'g')  # 小画布
```

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/output_80_1.png)

除此之外， 还有一种方法增加画布，就是plt.subplots()

```python
fig, axes = plt.subplots(nrows=1, ncols=2)  # 子图为 1 行，2 列
for ax in axes:
    ax.plot(x, y, 'r')
```

![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/output_86_0.png)

即便是只画一个画布，也建议通过fig,axses=plt.subplots()来生成画布和画板，方便调节，而不是使用plt.plot()

## 基础样式调整

### 添加图标题、图例

绘制包含图标题、坐标轴标题以及图例的图形，举例如下：

```python
fig, axes = plt.subplots()

axes.set_xlabel('x label')  # 横轴名称
axes.set_ylabel('y label')
axes.set_title('title')  # 图形名称

axes.plot(x, x**2)
axes.plot(x, x**3)
axes.legend(["y = x**2", "y = x**3"], loc=0)  # 图例
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_98_1.png)

图例中的 `loc` 参数标记图例位置，`1，2，3，4` 依次代表：右上角、左上角、左下角，右下角；`0` 代表自适应

### 线型、颜色、透明度

在 Matplotlib 中，你可以设置线的颜色、透明度等其他属性。

```python
fig, axes = plt.subplots()

axes.plot(x, x+1, color="red", alpha=0.5)
axes.plot(x, x+2, color="#1155dd")
axes.plot(x, x+3, color="#15cc55")
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_103_1.png)

而对于线型而言，除了实线、虚线之外，还有很多丰富的线型可供选择。

```python
fig, ax = plt.subplots(figsize=(12, 6))

# 线宽
ax.plot(x, x+1, color="blue", linewidth=0.25)
ax.plot(x, x+2, color="blue", linewidth=0.50)
ax.plot(x, x+3, color="blue", linewidth=1.00)
ax.plot(x, x+4, color="blue", linewidth=2.00)

# 虚线类型
ax.plot(x, x+5, color="red", lw=2, linestyle='-')
ax.plot(x, x+6, color="red", lw=2, ls='-.')
ax.plot(x, x+7, color="red", lw=2, ls=':')

# 虚线交错宽度
line, = ax.plot(x, x+8, color="black", lw=1.50)
line.set_dashes([5, 10, 15, 10])

# 符号
ax.plot(x, x + 9, color="green", lw=2, ls='--', marker='+')
ax.plot(x, x+10, color="green", lw=2, ls='--', marker='o')
ax.plot(x, x+11, color="green", lw=2, ls='--', marker='s')
ax.plot(x, x+12, color="green", lw=2, ls='--', marker='1')

# 符号大小和颜色
ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
ax.plot(x, x+15, color="purple", lw=1, ls='-',
        marker='o', markersize=8, markerfacecolor="red")
ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8,
        markerfacecolor="yellow", markeredgewidth=2, markeredgecolor="blue")

```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_106_1.png)

### 画布网格、坐标轴范围

有些时候，我们可能需要显示画布网格或调整坐标轴范围。设置画布网格和坐标轴范围。这里，我们通过指定 `axes[0]` 序号，来实现子图的自定义顺序排列。

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 显示网格
axes[0].plot(x, x**2, x, x**3, lw=2)
axes[0].grid(True)

# 设置坐标轴范围
axes[1].plot(x, x**2, x, x**3)
axes[1].set_ylim([0, 60])
axes[1].set_xlim([2, 5])
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_110_1.png)

除了折线图，Matplotlib 还支持绘制散点图、柱状图等其他常见图形。下面，我们绘制由散点图、梯步图、条形图、面积图构成的子图。

```python
n = np.array([0, 1, 2, 3, 4, 5])

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

axes[0].scatter(x, x + 0.25*np.random.randn(len(x)))
axes[0].set_title("scatter")

axes[1].step(n, n**2, lw=2)
axes[1].set_title("step")

axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")

axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5)
axes[3].set_title("fill_between")
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_113_1.png)

### 图形标注方法

当我们绘制一些较为复杂的图像时，阅读对象往往很难全面理解图像的含义。而此时，图像标注往往会起到画龙点睛的效果。图像标注，就是在画面上添加文字注释、指示箭头、图框等各类标注元素。

Matplotlib 中，文字标注的方法由 `matplotlib.pyplot.text()` 实现。最基本的样式为 `matplotlib.pyplot.text(x, y, s)`，其中 x, y 用于标注位置定位，s 代表标注的字符串。除此之外，你还可以通过 `fontsize=` , `horizontalalignment=` 等参数调整标注字体的大小，对齐样式等。

下面，我们举一个对柱形图进行文字标注的示例。

```python
fig, axes = plt.subplots()

x_bar = [10, 20, 30, 40, 50]  # 柱形图横坐标
y_bar = [0.5, 0.6, 0.3, 0.4, 0.8]  # 柱形图纵坐标
bars = axes.bar(x_bar, y_bar, color='blue', label=x_bar, width=2)  # 绘制柱形图
for i, rect in enumerate(bars):
    x_text = rect.get_x()  # 获取柱形图横坐标
    y_text = rect.get_height() + 0.01  # 获取柱子的高度并增加 0.01
    plt.text(x_text, y_text, '%.1f' % y_bar[i])  # 标注文字
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_119_0.png)

除了文字标注之外，还可以通过 `matplotlib.pyplot.annotate()` 方法向图像中添加箭头等样式标注。接下来，我们向上面的例子中增添一行增加箭头标记的代码。

```python
fig, axes = plt.subplots()

bars = axes.bar(x_bar, y_bar, color='blue', label=x_bar, width=2)  # 绘制柱形图
for i, rect in enumerate(bars):
    x_text = rect.get_x()  # 获取柱形图横坐标
    y_text = rect.get_height() + 0.01  # 获取柱子的高度并增加 0.01
    plt.text(x_text, y_text, '%.1f' % y_bar[i])  # 标注文字

    # 增加箭头标注
    plt.annotate('Min', xy=(32, 0.3), xytext=(36, 0.3),
                 arrowprops=dict(facecolor='black', width=1, headwidth=7))
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_122_0.png)

上面的示例中，`xy=()` 表示标注终点坐标，`xytext=()` 表示标注起点坐标。在箭头绘制的过程中，`arrowprops=()` 用于设置箭头样式，`facecolor=` 设置颜色，`width=` 设置箭尾宽度，`headwidth=` 设置箭头宽度，可以通过 `arrowstyle=` 改变箭头的样式。

## cheatsheet

matplotlib官方提供了cheatsheets，建议打印下来贴在显眼的地方

https://github.com/matplotlib/cheatsheets

## 三维图形绘制

### 基础三维图形

前面，我们已经了解了如果使用 Matplotlib 中的 pyplot 模块绘制简单的 2D 图像。其实，Matplotlib 也可以绘制 3D 图像，与二维图像不同的是，绘制三维图像主要通过 `mplot3d` 模块实现。但是，使用 Matplotlib 绘制三维图像实际上是在二维画布上展示，所以一般绘制三维图像时，同样需要载入 `pyplot` 模块。

`mplot3d` 模块下主要包含 4 个大类，分别是：

- `mpl_toolkits.mplot3d.axes3d()`
- `mpl_toolkits.mplot3d.axis3d()`
- `mpl_toolkits.mplot3d.art3d()`
- `mpl_toolkits.mplot3d.proj3d()`

其中，`axes3d()` 下面主要包含了各种实现绘图的类和方法。`axis3d()` 主要是包含了和坐标轴相关的类和方法。`art3d()` 包含了一些可将 2D 图像转换并用于 3D 绘制的类和方法。`proj3d()` 中包含一些零碎的类和方法，例如计算三维向量长度等。

一般情况下，我们用到最多的就是 `mpl_toolkits.mplot3d.axes3d()` 下面的 `mpl_toolkits.mplot3d.axes3d.Axes3D()` 类，而 `Axes3D()` 下面又存在绘制不同类型 3D 图的方法。

下面，我们通过几组示例，来学习 Matplotlib 绘制三维图形。首先，是三维散点图的绘制。

```python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline

# x, y, z 均为 0 到 1 之间的 100 个随机数
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)

fig = plt.figure()

ax = Axes3D(fig)
ax.scatter(x, y, z)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_14_1.png)

三维图形和二维图形在数据上的区别在于，三维图形多了一组数据用于度量多出来的一个维度。

当我们在桌面环境中绘制 3D 图形时，是可以通过鼠标任意拖动角度的，但在 Jupyter Notebook 环境中不支持，只会展示三维图形的默认视角静态图像。

线形图和散点图相似，需要传入 x,y,z*x*,*y*,*z* 三个坐标的数值。详细的代码如下。

```python
# 生成数据
x = np.linspace(-6 * np.pi, 6 * np.pi, 1000)
y = np.sin(x)
z = np.cos(x)

# 创建 3D 图形对象
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(x, y, z)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_19_1_1.png)

绘制完线型图，我们继续尝试绘制三维柱状图，其实它的绘制步骤和上面同样非常相似。

```python
# 创建 3D 图形对象
fig = plt.figure()
ax = Axes3D(fig)

# 生成数据并绘图
x = [0, 1, 2, 3, 4, 5, 6]
for i in x:
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    z = abs(np.random.normal(1, 10, 10))
    ax.bar(y, z, i, zdir='y', color=['r', 'g', 'b', 'y'])
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_22_0.png)

接下来需要绘制的三维曲面图要麻烦一些，我们需要对数据进行矩阵处理。其实和画二维等高线图很相似，只是多增加了一个维度。

```python
# 创建 3D 图形对象
fig = plt.figure()
ax = Axes3D(fig)

# 生成数据
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X ** 2 + Y ** 2)

# 绘制曲面图，并使用 cmap 着色
ax.plot_surface(X, Y, Z, cmap=plt.cm.winter)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_25_1_1.png)

`cmap=plt.cm.winter` 表示采用了 `winter` 配色方案。除了通过 `Axes3D()` 声明三维图形，我们也可以通过 `projection='3d'` 参数声明 3D 图形。

```python
fig = plt.figure(figsize=(14, 6))

# 通过 projection='3d' 声明绘制 3D 图形
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, Z, cmap=plt.cm.winter)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_28_1.png)

### 三维混合图

混合图就是将两种不同类型的图绘制在一张图里。绘制混合图一般有前提条件，那就是两种不同类型图的范围大致相同，否则将会出现严重的比例不协调，而使得混合图失去意义。

```
# 创建 3D 图形对象
fig = plt.figure()
ax = Axes3D(fig)

# 生成数据并绘制图 1
x1 = np.linspace(-3 * np.pi, 3 * np.pi, 500)
y1 = np.sin(x1)
ax.plot(x1, y1, zs=0, c='red')

# 生成数据并绘制图 2
x2 = np.random.normal(0, 1, 100)
y2 = np.random.normal(0, 1, 100)
z2 = np.random.normal(0, 1, 100)
ax.scatter(x2, y2, z2)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_32_1.png)

### 三维子图

我们已经学习过二维子图的绘制，其实三维情况下也是一样的。我们可以将二维图像和三维图像绘制在一起，又或者将几个三维图像绘制在一起。这里我们就拿上面绘制过的线形图和曲面图为例，看一看需要增删哪些代码。

```python
# 创建 1 张画布
fig = plt.figure(figsize=(8, 4))

# 向画布添加子图 1
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# 生成子图 1 数据
x = np.linspace(-6 * np.pi, 6 * np.pi, 1000)
y = np.sin(x)
z = np.cos(x)
# 绘制第 1 张图
ax1.plot(x, y, z)

# 向画布添加子图 2
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# 生成子图 2 数据
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X ** 2 + Y ** 2)
# 绘制第 2 张图
ax2.plot_surface(X, Y, Z, cmap=plt.cm.winter)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/matplotlib-basic/output_36_1.png)

三维图形的绘制，实际上是二维图形的衍生。在绘制方法上并无较大差别，你需要组织合适的数据，并声明三维绘图对象即可。

以上出自https://huhuhang.com/post/machine-learning/matplotlib-basic

# seaborn

Seaborn 基于 Matplotlib 核心库进行了更高阶的 API 封装，可以让你轻松地画出更漂亮的图形。Seaborn 的漂亮主要体现在配色更加舒服、以及图形元素的样式更加细腻，下面是 Seaborn 官方给出的参考图。

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/document-uid214893labid3264timestamp1501118752821.jpg)

## 快速优化图形

当我们使用 Matplotlib 绘图时，默认的图像样式算不上美观。此时，就可以使用 Seaborn 完成快速优化。下面，我们先使用 Matplotlib 绘制一张简单的图像。

```python
import matplotlib.pyplot as plt
%matplotlib inline

x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
y_bar = [3, 4, 6, 8, 9, 10, 9, 11, 7, 8]
y_line = [2, 3, 5, 7, 8, 9, 8, 10, 6, 7]

plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_11_1.png)

使用 Seaborn 完成图像快速优化的方法非常简单。只需要将 Seaborn 提供的样式声明代码 `sns.set()` 放置在绘图前即可。

```python
import seaborn as sns

sns.set()  # 声明使用 Seaborn 样式

plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_15_1.png)

我们可以发现，相比于 Matplotlib 默认的纯白色背景，Seaborn 默认的浅灰色网格背景看起来的确要细腻舒适一些。而柱状图的色调、坐标轴的字体大小也都有一些变化。

`sns.set()` 的默认参数为：

```python
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
```

其中：

- `context=''` 参数控制着默认的画幅大小，分别有 `{paper, notebook, talk, poster}` 四个值。其中，`poster > talk > notebook > paper`。
- `style=''` 参数控制默认样式，分别有 `{darkgrid, whitegrid, dark, white, ticks}`，你可以自行更改查看它们之间的不同。
- `palette=''` 参数为预设的调色板。分别有 `{deep, muted, bright, pastel, dark, colorblind}` 等，你可以自行更改查看它们之间的不同。
- 剩下的 `font=''` 用于设置字体，`font_scale=` 设置字体大小，`color_codes=` 不使用调色板而采用先前的 `'r'` 等色彩缩写。

## Seaborn 绘图 API

Seaborn 一共拥有 50 多个 API 类，相比于 Matplotlib 数千个的规模，可以算作是短小精悍了。其中，根据图形的适应场景，Seaborn 的绘图方法大致分类 6 类，分别是：关联图、类别图、分布图、回归图、矩阵图和组合图。而这 6 大类下面又包含不同数量的绘图函数。

接下来，我们就通过实际数据进行演示，使用 Seaborn 绘制不同适应场景的图形。

## 关联图

当我们需要对数据进行关联性分析时，可能会用到 Seaborn 提供的以下几个 API。

| 关联性分析  |       介绍       |
| :---------: | :--------------: |
|   relplot   |    绘制关系图    |
| scatterplot | 多维度分析散点图 |
|  lineplot   | 多维度分析线形图 |

[`relplot`](https://seaborn.pydata.org/generated/seaborn.relplot.html) 是 relational plots 的缩写，其可以用于呈现数据之后的关系，主要有散点图和条形图 2 种样式。我们载入鸢尾花示例数据集。

在绘图之前，先熟悉一下 iris 鸢尾花数据集。数据集总共 150 行，由 5 列组成。分别代表：萼片长度、萼片宽度、花瓣长度、花瓣宽度、花的类别。其中，前四列均为数值型数据，最后一列花的分类为三种，分别是：Iris Setosa、Iris Versicolour、Iris Virginica。

```python
iris = sns.load_dataset("iris")
iris.head()
```

| sepal_length | sepal_width | petal_length | petal_width | species |        |
| ------------ | ----------- | ------------ | ----------- | ------- | ------ |
| 0            | 5.1         | 3.5          | 1.4         | 0.2     | setosa |
| 1            | 4.9         | 3.0          | 1.4         | 0.2     | setosa |
| 2            | 4.7         | 3.2          | 1.3         | 0.2     | setosa |
| 3            | 4.6         | 3.1          | 1.5         | 0.2     | setosa |
| 4            | 5.0         | 3.6          | 1.4         | 0.2     | setosa |

此时，我们指定 x*x* 和 y*y* 的特征，默认可以绘制出散点图。

```python
sns.relplot(x="sepal_length", y="sepal_width", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_33_1.png)

但是，上图并不能看出数据类别之间的联系，如果我们加入类别特征对数据进行着色，就更加直观了。

```python
sns.relplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_36_1.png)

Seaborn 的函数都有大量实用的参数，例如我们指定 `style` 参数可以赋予不同类别的散点不同的形状。更多的参数，希望大家通过阅读官方文档了解。

```python
sns.relplot(x="sepal_length", y="sepal_width",
            hue="species", style="species", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_42_2.png)

你会发现，上面我们一个提到了 3 个 API，分别是：`relplot`，[`scatterplot`](https://seaborn.pydata.org/generated/seaborn.scatterplot.html) 和 [`lineplot`](https://seaborn.pydata.org/generated/seaborn.lineplot.html)。实际上，你可以把我们已经练习过的 `relplot` 看作是 `scatterplot` 和 `lineplot` 的结合版本。

这里就要提到 Seaborn 中的 API 层级概念，Seaborn 中的 API 分为 Figure-level 和 Axes-level 两种。`relplot` 就是一个 Figure-level 接口，而 `scatterplot` 和 `lineplot` 则是 Axes-level 接口。

Figure-level 和 Axes-level API 的区别在于，Axes-level 的函数可以实现与 Matplotlib 更灵活和紧密的结合，而 Figure-level 则更像是「懒人函数」，适合于快速应用。

例如上方的图，我们也可以使用 `lineplot` 函数绘制，你只需要取消掉 `relplot` 中的 `kind` 参数即可。

```python
sns.lineplot(x="sepal_length", y="petal_length",
             hue="species", style="species", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_48_2.png)

## 类别图

与关联图相似，类别图的 Figure-level 接口是 `catplot`，其为 categorical plots 的缩写。而 `catplot` 实际上是如下 Axes-level 绘图 API 的集合：

- 分类散点图：[`stripplot()`](https://seaborn.pydata.org/generated/seaborn.stripplot.html) (`kind="strip"`)[`swarmplot()`](https://seaborn.pydata.org/generated/seaborn.swarmplot.html) (`kind="swarm"`)
- 分类分布图：[`boxplot()`](https://seaborn.pydata.org/generated/seaborn.boxplot.html) (`kind="box"`)[`violinplot()`](https://seaborn.pydata.org/generated/seaborn.violinplot.html) (`kind="violin"`)[`boxenplot()`](https://seaborn.pydata.org/generated/seaborn.boxenplot.html) (`kind="boxen"`)
- 分类估计图：[`pointplot()`](https://seaborn.pydata.org/generated/seaborn.pointplot.html) (`kind="point"`)[`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) (`kind="bar"`)[`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) (`kind="count"`)

下面，我们看一下 `catplot` 绘图效果。该方法默认是绘制 `kind="strip"` 散点图。

```python
sns.catplot(x="sepal_length", y="species", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_54_1.png)

`kind="swarm"` 可以让散点按照 beeswarm 的方式防止重叠，可以更好地观测数据分布。

```python
sns.catplot(x="sepal_length", y="species", kind="swarm", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_56_1.png)

同理，`hue=` 参数可以给图像引入另一个维度，由于 iris 数据集只有一个类别列，我们这里就不再添加 `hue=` 参数了。如果一个数据集有多个类别，`hue=` 参数就可以让数据点有更好的区分。

接下来，我们依次尝试其他几种图形的绘制效果。绘制箱线图：

```python
sns.catplot(x="sepal_length", y="species", kind="box", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_59_1.png)

绘制小提琴图：

```python
sns.catplot(x="sepal_length", y="species", kind="violin", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_62_2.png)

绘制增强箱线图：

```python
sns.catplot(x="species", y="sepal_length", kind="boxen", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_65_1.png)

绘制点线图：

```python
sns.catplot(x="sepal_length", y="species", kind="point", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_68_2.png)

绘制条形图：

```python
sns.catplot(x="sepal_length", y="species", kind="bar", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_71_2.png)

## 分布图

分布图主要是用于可视化变量的分布情况，一般分为单变量分布和多变量分布。当然这里的多变量多指二元变量，更多的变量无法绘制出直观的可视化图形。

Seaborn 提供的分布图绘制方法一般有这几个：[`jointplot`](https://seaborn.pydata.org/generated/seaborn.jointplot.html)，[`pairplot`](https://seaborn.pydata.org/generated/seaborn.pairplot.html)，[`distplot`](https://seaborn.pydata.org/generated/seaborn.distplot.html)，[`kdeplot`](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)。接下来，我们依次来看一下这些绘图方法的使用。

Seaborn 快速查看单变量分布的方法是 `distplot`。默认情况下，该方法将会绘制直方图并拟合核密度估计图。

```python
sns.distplot(iris["sepal_length"])
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_80_2.png)

`istplot` 提供了参数来调整直方图和核密度估计图，例如设置 `kde=False` 则可以只绘制直方图，或者 `hist=False` 只绘制核密度估计图。当然，`kdeplot` 可以专门用于绘制核密度估计图，其效果和 `distplot(hist=False)` 一致，但 `kdeplot` 拥有更多的自定义设置。

```python
sns.kdeplot(iris["sepal_length"])
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_83_2.png)

`jointplot` 主要是用于绘制二元变量分布图。例如，我们探寻 `sepal_length` 和 `sepal_width`二元特征变量之间的关系。

```python
sns.jointplot(x="sepal_length", y="sepal_width", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_86_2.png)

`jointplot` 并不是一个 Figure-level 接口，但其支持 `kind=` 参数指定绘制出不同样式的分布图。例如，绘制出核密度估计对比图。

```python
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="kde")
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_89_2.png)

六边形计数图：

```python
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="hex")
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_92_2.png)

回归拟合图：

```python
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="reg")
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_95_2.png)

最后要介绍的 `pairplot` 更加强大，其支持一次性将数据集中的特征变量两两对比绘图。默认情况下，对角线上是单变量分布图，而其他则是二元变量分布图。

```python
sns.pairplot(iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_98_1.png)

此时，我们引入第三维度 `hue="species"` 会更加直观。

```python
sns.pairplot(iris, hue="species")
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_101_2.png)

## 回归图

接下来，我们继续介绍回归图，回归图的绘制函数主要有：[`lmplot`](https://seaborn.pydata.org/generated/seaborn.lmplot.html) 和 [`regplot`](https://seaborn.pydata.org/generated/seaborn.regplot.html)。

`regplot` 绘制回归图时，只需要指定自变量和因变量即可，`regplot` 会自动完成线性回归拟合。

```python
sns.regplot(x="sepal_length", y="sepal_width", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_106_2.png)

`lmplot` 同样是用于绘制回归图，但 `lmplot` 支持引入第三维度进行对比，例如我们设置 `hue="species"`。

```python
sns.lmplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_109_2.png)

## 矩阵图

矩阵图中最常用的就只有 2 个，分别是：[`heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html) 和 [`clustermap`](https://seaborn.pydata.org/generated/seaborn.clustermap.html)。

意如其名，`heatmap` 主要用于绘制热力图。

```python
import numpy as np

sns.heatmap(np.random.rand(10, 10))
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_114_1.png)

热力图在某些场景下非常实用，例如绘制出变量相关性系数热力图。

除此之外，`clustermap` 支持绘制层次聚类结构图。如下所示，我们先去掉原数据集中最后一个目标列，传入特征数据即可。当然，你需要对层次聚类有所了解，否则很难看明白图像多表述的含义。

```python
iris.pop("species")
sns.clustermap(iris)
```

![img](https://cdn.jsdelivr.net/gh/huhuhang/cdn/images/2019/seaborn-basic/output_118_1.png)

如果你浏览官方文档，你会发现 Seaborn 中还存在大量已大些字母开始的类，例如 `JointGrid`，`PairGrid` 等。实际上这些类只是其对应小写字母的函数 `jointplot`，`pairplot` 的进一步封装。当然，二者可能稍有不同，但并没有本质的区别。

除此之外，[Seaborn 官方文档](https://seaborn.pydata.org/api.html) 中还有关于 [样式控制](https://seaborn.pydata.org/api.html#style-control) 和 [色彩自定义](https://seaborn.pydata.org/api.html#color-palettes) 等一些辅助组件的介绍。对于这些 API 的应用没有太大的难点，重点需要勤于练习。

来自https://huhuhang.com/post/machine-learning/seaborn-basic



