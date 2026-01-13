---
title: GOCI-II data is available now
tags:
  - Research
  - Ocean Color
  - Oceanography
categories: 学习笔记
abbrlink: 61b56df3
date: 2021-10-13 09:34:06
mathjax:
copyright:
password:
---

GOCI-II是韩国发布的第二代海洋水色遥感卫星，覆盖几乎整个东亚和东南亚，现在产品已经初步开放。

<!-- more -->

# 下载网址

这里是L2http://www.khoa.go.kr/nosc/satellite/searchL2.do的下载地址，不过可惜是韩语的。开个谷歌翻译大概也能看懂。

![1634089418039](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1634089418039.png)

选取想要的时间之后，点击NETCDF左边的灰色按钮就是下载

# 基本信息

![1634089702973](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1634089702973.png)

基本的产品包括这些，该有的都有，和GOCI一样提供了Rho。

L2产品空间分辨率300m，时间分辨率一小时，覆盖范围如图（来自英语版ATBD）

![1634090002375](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1634090002375.png)

基本波段特征如图(来自ATBD)

![1634090044764](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1634090044764.png)

![1634090071333](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1634090071333.png)

红波段有三个，而且有709。



# ATBD

这里是英语版的ATBDhttp://www.khoa.go.kr/nosc/satellite/searchL2.do，捡了几个重要的看一下。

## 结果检验

目前都没有用实地数据做检测，都是模拟数据集。

## 大气校正

基本流程如图

![1634090342354](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1634090342354.png)

其中NIR气溶胶反射率的计算如下图

![1634090229832](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1634090229832.png)

我记得前不久他们团队还发了一篇近岸NIR的两个模型的结果在remote sensing上，有兴趣可以去看看。

基本的流程和传统算法差不多，都是假定近红外区域水体反射率为０，然后外推到可见光区域。这里增加了一个近红外在近岸区域水体反射率的经验矫正，然后进行反复迭代，通过阈值判断是否完成大气校正。

之前记录了何老师在水色班的[PPT](https://lifeodyssey.github.io/posts/23b76a18.html?highlight=%E5%A4%A7%E6%B0%94)，相关的视频录屏在B站也可以找到

## IOP

用的是QAA

![1634090622389](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1634090622389.png)

自己之前发过[一篇文章](https://lifeodyssey.github.io/posts/f5ee8139.html?highlight=retr)用来讲这个

## Chl-a

用的Hu那个CI和OC融合的算法

![1634090811945](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1634090811945.png)

本来想下了数据看一下结构写个画图函数，结果上面那个网址在10.14无法访问了，等以后吧。

最近可能会更新个xarray相关的文章

## GOCI-||初期验证结果

前几天邮箱被轰炸了，然后打开一看发现是GOCI-||的初期验证机过发布了。详细请看Korean Journal of Remote Sensing, Vol. 37, No. 5, October, 2021.

虽然发表在韩语期刊上，但是大概还是能看懂得，挑几个关心的放一下。

![image-20211125163835678](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/202111251653097.png)

![image-20211125165443237](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/202111251654280.png)

这么看结果好像还行，很符合预期。

居然没有叶绿素啥的结果。
