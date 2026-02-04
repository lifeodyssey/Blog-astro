---
title: Cousera机器学习基石第四周笔记-Machine-Learning-Foundation-Week-4-Note-in-Cousera
tags:
  - 机器学习
  - Cousera
  - 林轩田
categories: 学习笔记
mathjax: true
typora-root-url: ..
abbrlink: c7f63b06
date: 2019-04-02 15:08:38
copyright: true
lang: en
---
**Feasibility of Learning**

<!-- more -->

# Learning is Impossible?



# Probability to the Rescue

## Inferring Something Unknow

in sample$\rightarrow$out sample

## Possible versus Probable

## Hoeffding’s Inequality

In big sample(*N* large),$\boldsymbol{\upsilon}$ is probably close to $\boldsymbol{u}$ (within $\boldsymbol{\epsilon}$)
$$
\mathbb{P}[|v-u|>\boldsymbol{\epsilon}]\leq2exp(-2\boldsymbol{\epsilon}^2N)
$$
called Hoeffding’s Inequality, for marbles,coin,polling

the statement $v=u$ is **probably approximately correct**(PAC)

- valid for all N and $\boldsymbol{\epsilon}$

- does not depend on $u$,no need to know$u$

- larger sample size *N* or looser gap $\boldsymbol{\epsilon} \rightarrow$higher probability for $v=u$

  if large N,can probably infer unknown $u$ by know $v$

  

# Connection  to Learning

## Added Components

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/微信截图_20190404101452.png)

for any fixed *h*, can probably infer unkown $E_out(h)=\underset{X\approx P}{\varepsilon}[h(x)\ne f(x)]$by known$ E_in(h)=\frac{1}{N}\sum^N_{n=1}[h(x)\ne f(x)]$

## The Formal Guarantee

if $E_{in}(h)\approx E_{out}(h)$ and$E_{in}(h)small\rightarrow E_{out}(h)samll\rightarrow h\approx f $with respect to P

## Verification of One h

if $E_{in}(h)$ small for the fixed h and A pick the h as g$\rightarrow$ g=f PAC

if A force to pick THE h as g$\rightarrow E_{in}(h)$ almost always not small$\rightarrow g\ne f$ PAC

real learning:

  A shall make choices$\in \H$ (like PLA) rather than being forced to pick one h.

## The ‘Verification’ Flow

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1554345523532.png)



# Connection to Real Learning

## BAD Sample and BAD Data

BAD Sample:$E_{out}=\frac{1}{2}$,but getting all heads($E_{in}=0$)

BAD Data for One h:$E_{out}(h)$ and $E_{in}h$ far away

## BAD data for Many h

BAD data for many h $\Leftrightarrow$ no freedom of choice by A $\Leftrightarrow$ there exists some h such that $E_{out}(h)$ and $E_{in}(h)$ far away

## Bound of BAD Data

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1554349806026.png)

## The Statistical Learning Flow

if $|\mathbb{H}|$= M finite, N large enough,for whatever g picked by A,$E_{out}(g)\approx E_{in}(g)$

if A finds one g with $E_{in}(g)\approx 0$,PAC guarantee for$E_{out}(g)\Rightarrow$learning possible

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/1554350153964.png)

M=$\infty$? - see you in the next lectures~

# 吐槽

这个作业题是真的难啊，花了一个半小时才堪堪通过，尤其是最后几个写PLA和pocket算法的