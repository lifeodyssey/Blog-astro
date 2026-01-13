---
title: Chla retrieval
tags:
  - Ocean Color
  - Chla Algorithm
categories: Annotated Bibliography
mathjax: true
abbrlink: a31ba57b
date: 2021-03-12 14:21:07
copyright:
---

A note for Chla algorithm.

<!-- more -->

# Chla Algorithms Review

| Name        | Formula                                                      | Reference                      | Water type               |
| ----------- | ------------------------------------------------------------ | ------------------------------ | ------------------------ |
| OC          | $log_{10}(Chl)=a_0+a_1X+a_2X^2+a_3X^3+a_4X^4,X=log_{10}(\frac{Rrs_(\lambda_b)}{Rrs_(\lambda_g)})$(Example) | O'Reilly 2019                  | Clear                    |
| OCI         | ![image-00030312152124019](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152124019.png) | Wang 2016;Hu 2011              | Coastal                  |
| Band ration | $Rrs(751)/Rrs(672)$                                          | Gurlin 2011                    | Coastal                  |
| Tang/Tassan | ![image-00030312152233703](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152233703.png) | Tang 2004;Tassan,1994;Sun 2010 | Coastal                  |
| Gitelson    | ![image-00030312152341386](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152341386.png) | Gitelson 2008                  | Coastal                  |
| RLH         | Rrs(672)-Rrs(555)-{[Rrs(751)-Rrs(555)] × [(672-555)/(751-555)]} | Schalles,1998                  | Coastal                  |
| RDR         | [Rrs(555)-Rrs(488)]/[Rrs(555) + Rrs(488)]                    | Gitelson,1993                  | Coastal                  |
| NDCI        | [Rrs(751)-Rrs(672)]/[Rrs(751) + Rrs(672)]                    | Mishra, 2012                   | Coastal                  |
| SCI         | HChla = {Rrs(751) + (751-672)/(751-555) × [Rrs(555)-Rrs(751)]}-Rrs(672), H∆ = Rrs(555)-{Rrs(751) + (751-555)/(751-488) × [Rrs(488)-Rrs(751)]}, SCI = HChla-H∆ | Shen, 2010                     | Coastal                  |
| ASA         | ![image-00030312152813412](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152813412.png) | Jiang, 2020                    | Coastal                  |
| TC2         | ![image-00030312152928782](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312152928782.png) | Liu,2020                       | Coastal, inversion based |
| NIR-Red     | ![image-00030312153111284](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312153111284.png) | Gons,2005                      | Cosatal                  |
| MDN         | NN                                                           | Pahlevan 2020                  | Hyperspectral,Coastal    |
| OWT         | ![image-00030312153858028](/Users/zhenjia/Library/Application Support/typora-user-images/image-00030312153858028.png) | Neil, 2019                     | Coastal                  |

There are a lot of Chla algorithm.

Possibly, it is almost out of date.

Just JO I think hhh.

## An interesting paper

Progressive scheme for blending empirical ocean color retrievals of absorption coefficient and chlorophyll concentration from open oceans to highly turbid waters

Former Classification scheme is based on value or shape. It is a hard transaction, not smoothly.

One way is use fuzzy c-mean, given隶属度(MD，这还是这么久了我第一次见到不会用英语的专业名词).But it is statistically based, don't have any implicit physics.

This is a paper have implicit physics.

