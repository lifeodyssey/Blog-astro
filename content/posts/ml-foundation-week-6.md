---
title: Cousera机器学习基石第六周笔记 Machine Learning Foundation Week 6 Note in Cousera
copyright: true
tags:
  - 机器学习
  - Cousera
  - 林轩田
categories: 学习笔记
mathjax: true
abbrlink: da4d42a2
date: 2019-05-12 15:44:17
---

Theory of Generalization

test error can approximate training error if there is enough data and growth function does not grow too fast


  

<!-- more -->

# Restriction of Breaking Point

## The Four Breaking Points

growth function $m_\H(N)$:max number of dichotomies

- positive rays: $m_\H(N)=N+1$
- positive intervals: $m_\H(N)=\frac{1}{2}N^2+\frac{1}{2}N+1$
- convex sets: $m_\H(N)=2^N$
- 2D perceptrons :$m_\H(N)<2^N$ in some cases

## Restriction of Breaking Point

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20190512171932.png)



# Bounding Function:Basic Cases

## Bounding Function

