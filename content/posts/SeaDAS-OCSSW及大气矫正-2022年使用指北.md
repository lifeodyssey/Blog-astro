---
title: SeaDAS OCSSW及大气矫正 2022年使用指北
copyright: true
tags:
  - Research
  - Ocean Color
  - Oceanography
  - MODIS
  - 大气校正
categories: 学习笔记
mathjax: true
abbrlink: 182a5f48
date: 2022-02-11 20:09:56
---

最近越来越多小伙伴顺着[这个](https://lifeodyssey.github.io/posts/ce08f3a2.html)来找到我，特此更新一篇。

欢迎任何问题的交流和咨询 zhenjiazhou0127@outlook.com

<!-- more -->

# 一个简单的大气矫正方法个人推荐

1. 如果你要做瑞利校正，可以和我一起来开发测试[这个](https://github.com/lifeodyssey/RayleighCorrection)基于py6s的包，目前尚在缓慢开发测试中（懒）。或者选择用SeaDAS OCSSW
2. 如果你不做瑞利校正，做全部的大气校正，那么你有如下几个选择，[OC-SMART](http://www.rtatmocn.com/oc-smart/), [acolite](https://github.com/acolite/acolite), snap，SeaDAS OCSSW和GDPS。这其中无论您需要处理哪一颗卫星的数据，我都非常建议您尝试使用OC-SMART。以下是不包括OC-SMART的推荐
3. 如果你需要用到landsat或者sentinel MSI,可以尝试 [acolite](https://github.com/acolite/acolite)。
4. 如果你需要用到GOCI，那么我推荐GDPS
5. 如果你需要用到OLCI，那么我推荐SNAP或者SeaDAS OCSSW
6. 如果你要用ＭODIS,VIIRS，那么可以尝试SeaDAS

# 我个人的使用体验

注意这里不包括对于精确度的评价。

| 名称         | 安装设置难度 | 计算速度 | 批处理使用难度                              | 编程要求 |
| ------------ | ------------ | -------- | ------------------------------------------- | -------- |
| SeaDAS OCSSW | 最难         | 非常慢   | 比较麻烦                                    | 无       |
| SNAP         | 不难         | 比较慢   | 没用过...                                   | 无       |
| GDPS         | 不难         | 比较慢   | 不支持，另外仅支持GOCI，MODIS等少数几种数据 | 无       |
| OC-SMART     | 不需要安装   | 快       | 简单                                        | 高       |
| ACOLITE      | 不需要安装   | 快       | 简单                                        | 中等     |

# 如果你非要使用SeaDAS OCSSW

那么我不推荐你在虚拟机里使用，建议参考[这个](https://lifeodyssey.github.io/posts/eff06e8d.html)和[这个](https://lifeodyssey.github.io/posts/60b81fb7.html)在远程服务器或者WSL上安装OCSSW，会方便很多。

