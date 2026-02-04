---
title: クロロフィルa推定アルゴリズム
tags:
  - Ocean Color
  - Chla Algorithm
categories: 文献ノート
mathjax: true
abbrlink: a31ba57b
date: 2021-03-12 14:21:07
copyright:
lang: ja
---

クロロフィルaアルゴリズムのノート。

<!-- more -->

# クロロフィルaアルゴリズムレビュー

| 名称        | 公式                                                      | 参考文献                      | 水域タイプ               |
| ----------- | ------------------------------------------------------------ | ------------------------------ | ------------------------ |
| OC          | $log_{10}(Chl)=a_0+a_1X+a_2X^2+a_3X^3+a_4X^4,X=log_{10}(\frac{Rrs_(\lambda_b)}{Rrs_(\lambda_g)})$(例) | O'Reilly 2019                  | 透明水域                    |
| OCI         | ![image-00030312152124019](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152124019.png) | Wang 2016;Hu 2011              | 沿岸水域                  |
| Band ration | $Rrs(751)/Rrs(672)$                                          | Gurlin 2011                    | 沿岸水域                  |
| Tang/Tassan | ![image-00030312152233703](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152233703.png) | Tang 2004;Tassan,1994;Sun 2010 | 沿岸水域                  |
| Gitelson    | ![image-00030312152341386](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152341386.png) | Gitelson 2008                  | 沿岸水域                  |
| RLH         | Rrs(672)-Rrs(555)-{[Rrs(751)-Rrs(555)] × [(672-555)/(751-555)]} | Schalles,1998                  | 沿岸水域                  |
| RDR         | [Rrs(555)-Rrs(488)]/[Rrs(555) + Rrs(488)]                    | Gitelson,1993                  | 沿岸水域                  |
| NDCI        | [Rrs(751)-Rrs(672)]/[Rrs(751) + Rrs(672)]                    | Mishra, 2012                   | 沿岸水域                  |
| SCI         | HChla = {Rrs(751) + (751-672)/(751-555) × [Rrs(555)-Rrs(751)]}-Rrs(672), H∆ = Rrs(555)-{Rrs(751) + (751-555)/(751-488) × [Rrs(488)-Rrs(751)]}, SCI = HChla-H∆ | Shen, 2010                     | 沿岸水域                  |
| ASA         | ![image-00030312152813412](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152813412.png) | Jiang, 2020                    | 沿岸水域                  |
| TC2         | ![image-00030312152928782](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152928782.png) | Liu,2020                       | 沿岸水域、逆推定ベース |
| NIR-Red     | ![image-00030312153111284](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312153111284.png) | Gons,2005                      | 沿岸水域                  |
| MDN         | NN                                                           | Pahlevan 2020                  | ハイパースペクトル、沿岸水域    |
| OWT         | ![image-00030312153858028](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312153858028.png) | Neil, 2019                     | 沿岸水域                  |

クロロフィルaアルゴリズムはたくさんあります。

おそらく、もう少し古くなっているかもしれません。

ただのメモです笑。

## 興味深い論文

Progressive scheme for blending empirical ocean color retrievals of absorption coefficient and chlorophyll concentration from open oceans to highly turbid waters

以前の分類スキームは値または形状に基づいていました。これはハードな遷移であり、スムーズではありません。

一つの方法はファジーc平均を使用し、帰属度（MD）を与えることです。しかし、これは統計的に基づいており、暗黙の物理的意味を持っていません。

これは暗黙の物理的意味を持つ論文です。
