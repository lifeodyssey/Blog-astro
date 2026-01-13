---
title: Cousera机器学习基石第五周笔记-Machine-Learning-Foundation-Week-5-Note-in-Cousera
tags:
  - 机器学习
  - Cousera
  - 林轩田
categories:
  - 学习笔记
mathjax: true
abbrlink: fbd0b1b0
date: 2019-04-06 13:23:00
copyright: true
---
**Traning versus Testing**

Actually,I didn’t fully understand this part of course.However  I don’t focus on the theory but application.

I will update this note if possible.

<!-- more -->

# Recap and Preview

## Recap: The ‘Statistical’ Learning Flow

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1554528606498.png)


## Two Central Questions

1\. can we make sure that $E_{out}(g)$ is close enough to $E_{in}(g)$?

2\. can we make $E_{in}(g)$ small enough?

$\Rightarrow$ What role does $\underbrace{M}\\{|\mathscr{H}|}$ play for the two questions?

##　Trade-off on *M

| small M                                             | large M                                            |
| --------------------------------------------------- | -------------------------------------------------- |
| 1\. Yes $\mathbb{P}[BAD]\leq2\cdot M\cdot exp(...)$ | 1\. No!$\mathbb{P}[BAD]\leq2\cdot M\cdot exp(...)$ |
| 2\. $\color{Maroon}No!$,two few choices             | 2\. Yes!many choices                               |

using the right M is important

## Preview

- Know: $\mathbb{P}[|E_{in}(g)-E_{out}(g)|>\epsilon]\leq 2 \cdot M\cdot exp(-2\epsilon^2N)$
- To do 
  - establish a finite quantity that replaces M$\mathbb{P}[|E_{in}(g)-E_{out}(g)|>\epsilon]\leq 2 \cdot m_{\mathscr{H}}\cdot exp(-2\epsilon^2N)$
  - justify the feasibility of learning for infinite M
  - study $m_{\mathscr{H}} $to understand its trade-off for ‘right’$\mathscr{H}$,just like M

# Effective Number of Lines

## Where Did *M* Come From

The BAD event ：uniform bound fail$\rightarrow$ M$\approx\infty$

## Where Did Uniform Bound Fail

overlapping for similar hypotheses

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20190509150052.png)

union bound over-estimating

**To acoount for overlap,can we group similar hypotheses by kind?**

## How Many Lines Are There?

In 2D,we can get smaller than $2^N$lines

## Effective Number of Lines

maximun kind of lines with respect to N inputs $x_1$,$x_2$,.....,$x_N$

$\iff$==effctive number of lines==

In 2D,the effective number of lines

- must be $\leq 2^N$

- finite ‘grouping’ of infinitely-many lines $\in H$

- wish:

  $\mathbb{P}[|E_{in}(g)-E_{out}(g)|>\epsilon]\leq 2 \cdot effctive(N)\cdot exp(-2\epsilon^2N)$

If 

1. effective(N) can replace M and 

2. effective(N) $\ll 2^N$

   ==learning possible with infinitie lines==

# Effective Number of Hypotheses



## Dichotomies:Mini-hypotheses

# Break Points



