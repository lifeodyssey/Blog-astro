---
title: Cousera机器学习基石第二周笔记 Machine Learning Foundation Week 2 Note in Cousera
tags:
  - 机器学习
  - Cousera
  - 林轩田
categories: 学习笔记
mathjax: true
abbrlink: 2c9001a2
copyright: true
date: 2019-04-08 15:08:38
lang: en
---
Learning to Answer Yes/No
<!-- more -->

# Perceptron Hypothesis Set

Example:Credit Approval Problem Revisited

## A Simple Hypothesis Set:the 'Perceptron'

- For **x**=$(x_1,x_2,...,x_d)$ 'feature of customer',compute a weighted'score' and

  \<center\> $approve \quad credit\quad if \sum_{i=1}^{d} w_i x_i>threshold$\<center\>

  <center\> $deny \quad credit\quad if \sum_{i=1}^{d} w_i x_i<threshold$\<center\>

- $y:\{+1(good),-1(bad)\}$,0 ignored-linear formula$h\in H$are

  \<center\>$h(x)=sign((\sum_{i=i}^{d}w_i x_i)-threshold)$\<center\>

this is called'**perceptron**'hypothesis histroically

## Vector Form of Perceptron Hypothesis

$$
\begin{split}
h(x)&=sign((\sum_{i=1}^{d}w_i x_i)-threshold)\\
   
    &=sign((\sum_{i=1}^{d}w_i x_i)+\underbrace{(threshold)}_{w_0}\cdot\underbrace{(+1)}_{x_0})\\
   
    &=sign(sum_{i=0}^{d}w_i x_i)\\
    &=sign(W^T x)
\end{split}
$$

- each'tall'**w** represents a hypothesis *h* &is multiplied with 'tall' **x**——will use tall versions to simplify notation

Q:What does *h* look like

## Perceptrons in $R^2$

**perceptrons$\Leftrightarrow$linear(binary) classifiers**

# Perceptron Learning Algorithm (PLA)

## Select *g* from H

- want: g$\approx$f(hard when f unkown)
- almost necessary:g$\approx$f on D,ideally $g(x_n)=f(x_n)=y_n$
- difficult:H is of infinite size
- idea: start from some $g_0$,and 'correct' its mistakes on D

## Perceptron Learning Algorithm

For t=0,1,...

1\. find a mistake of $w_t$ called$(x_{n(t)},y_{n(t)})$

sign$(W^{T}_t X_{n(t)})\not=y_{n(t)}$

2\. (try to) correct the mistake by

$w_{t+1}\leftarrow w_t+y_{n(t)}x_{n(t)}$

...until no more mistakes 

return last **w** (called $w_{PLA}$ )as *g*

My question:why determinant is like vector

## Pratical implementation of PLA

### Cyclic PLA
## Some Remaning issues of PLA
###　Algorithmic:halt(no mistake)?
- naive cyclic:??
- random cyclic:??
- other variant:??

### Learning:$g\approx f$ ?
- on D, if halt,yes(no mistake)
- outside D:??
- if not halting:??

# Guarantee of PLA
## Linear Separability

- if PLA halts(i.e. no more mistakes),
(necessary condition)D allows some **w** to make no mistakes
- call such D linear separable

## PLA Fact: $w_t$Gets More Aligned with $w_f$

- $w_f$ perfect hence every $x_n$ correctly away from line:
  $y_{n(t)}w_f^T x_{n(t)}\ge \underset{n}{min}  y_{n(t)}w_f^T x_{n(t)}>0$

- $w_f^T w_t\uparrow$by updating with any$(x_{n(t)},y_{n(t)})$
  $$
  \begin{split}
  w_f^Tw_{t+1}&=w_f^T(w_t+y_{n(t)}x_{n(t)})\\
  &\ge w_f^Tw_{t}+\underset{n}{min}  y_{n(t)}w_f^T x_{n(t)}\\
  &> w_f^Tw_{t}+0
  \end{split}
  $$


$w_t$ appears more algned with $w_f$

Q:the length of vector

## PLA fact: $w_t$ Does Not Grow Too Fast



$w_t$ changed only when mistake

$\Leftrightarrow sign(w_t^Tx_{n(t)})\not=y_{n(t)}\Leftrightarrow y_{n(t)}w_t^Tx_{n(t)}\le 0$

- mistake ‘limits’$||w_t||^2$ growth,even when updating with’longest’ $x_n$
  $$
  \begin{split}
  ||w_{t+1}||^2&=||w_t+y_{n(t)}x_{n(t)}||^2\\
  &=||w_t||^2+2y_{n(t)}w_t^Tx_{n(t)}+||y_{n(t)}x_{n(t)}||^2\\
  &\le ||w_t||^2+0+||y_{n(t)}x_{n(t)}||^2\\
  &\le||w_t||^2+\underset{n}{max}||y_nx_n||^2
  \end{split}
  $$
  

start from $w_0=0$,after*T* mistake corrections,
$$
\frac{w_f^T\quad w_T}{||w_f||\quad||w_T||}\ge \sqrt{T}\cdot constant
$$

## Novikoff  theorem

ref:Lihang<statistic learning approach>page 31

设训练数据集*T*=${(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$是线性可分的，其中$x_i\in\mathcal{X}=R^n,y_i\in\mathcal{Y}={-1,+1},i=1,2,...,N$,则

（1）存在满足条件$||\hat{w}_{opt}||=1$的超平面$\hat{w}_{opt}\cdot\hat{x}+b_{opt}=0$将训练数据集完全正确分开，且存在$\gamma>0$，对所有$i=1,2,...,N$
$$
y_i(\hat{w}_{opt}\cdot\hat{x}_i)=y_i(\hat{w}_{opt}\cdot\hat{x}_i+b_{opt}\ge\gamma
$$

(2)令$R=\underset{1\le i\le N}{max}||\hat{x}_i||$,则感知机算法在训练数据集上的误分类次数*k*满足不等式
$$
k\le (\frac{R}{\gamma})^2
$$

# Non-Separable Data

## More about PLA

CONS:maybe not’linnear separable’,and not fully sure how long halting takes

## Learning with Noisy Data

## Line with Noise Tolerance

NP-hard

## Pocket Algorithm

Find the least mistakes until iterations

# Summary

- Perceptron Hypothesis Set

  hyperplanes/linear classifiers in $\mathcal{R}^d$

- Perceptron Learning Algorithm(PLA)

  correct mistakes and improve iteratively

- Guarantee of PLA

  no mistake eventually if linear separable

- Non-Separable Data

  hold somewhat’best’weights in pocket