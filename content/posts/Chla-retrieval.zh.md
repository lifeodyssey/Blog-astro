---
title: 叶绿素a反演算法
tags:
  - Ocean Color
  - Chla Algorithm
categories: 文献笔记
mathjax: true
abbrlink: a31ba57b
date: 2021-03-12 14:21:07
copyright:
lang: zh
---

叶绿素a算法笔记。

<!-- more -->

# 叶绿素a算法综述

| 名称        | 公式                                                      | 参考文献                      | 水体类型               |
| ----------- | ------------------------------------------------------------ | ------------------------------ | ------------------------ |
| OC          | $log_{10}(Chl)=a_0+a_1X+a_2X^2+a_3X^3+a_4X^4,X=log_{10}(\frac{Rrs_(\lambda_b)}{Rrs_(\lambda_g)})$(示例) | O'Reilly 2019                  | 清澈水体                    |
| OCI         | ![image-00030312152124019](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152124019.png) | Wang 2016;Hu 2011              | 近岸水体                  |
| Band ration | $Rrs(751)/Rrs(672)$                                          | Gurlin 2011                    | 近岸水体                  |
| Tang/Tassan | ![image-00030312152233703](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152233703.png) | Tang 2004;Tassan,1994;Sun 2010 | 近岸水体                  |
| Gitelson    | ![image-00030312152341386](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152341386.png) | Gitelson 2008                  | 近岸水体                  |
| RLH         | Rrs(672)-Rrs(555)-{[Rrs(751)-Rrs(555)] × [(672-555)/(751-555)]} | Schalles,1998                  | 近岸水体                  |
| RDR         | [Rrs(555)-Rrs(488)]/[Rrs(555) + Rrs(488)]                    | Gitelson,1993                  | 近岸水体                  |
| NDCI        | [Rrs(751)-Rrs(672)]/[Rrs(751) + Rrs(672)]                    | Mishra, 2012                   | 近岸水体                  |
| SCI         | HChla = {Rrs(751) + (751-672)/(751-555) × [Rrs(555)-Rrs(751)]}-Rrs(672), H∆ = Rrs(555)-{Rrs(751) + (751-555)/(751-488) × [Rrs(488)-Rrs(751)]}, SCI = HChla-H∆ | Shen, 2010                     | 近岸水体                  |
| ASA         | ![image-00030312152813412](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152813412.png) | Jiang, 2020                    | 近岸水体                  |
| TC2         | ![image-00030312152928782](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152928782.png) | Liu,2020                       | 近岸水体，基于反演 |
| NIR-Red     | ![image-00030312153111284](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312153111284.png) | Gons,2005                      | 近岸水体                  |
| MDN         | NN                                                           | Pahlevan 2020                  | 高光谱，近岸水体    |
| OWT         | ![image-00030312153858028](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312153858028.png) | Neil, 2019                     | 近岸水体                  |

叶绿素a算法有很多种。

可能已经有些过时了。

只是随便记录一下哈哈哈。

## 一篇有趣的论文

Progressive scheme for blending empirical ocean color retrievals of absorption coefficient and chlorophyll concentration from open oceans to highly turbid waters

以前的分类方案是基于数值或形状的。这是一种硬性转换，不够平滑。

一种方法是使用模糊c均值，给出隶属度（MD）。但这是基于统计的，没有任何隐含的物理意义。

这是一篇具有隐含物理意义的论文。
