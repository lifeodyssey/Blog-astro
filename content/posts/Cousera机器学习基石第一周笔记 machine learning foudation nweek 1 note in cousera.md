---
title: Cousera机器学习基石第一周笔记 Machine Learning Foundation Week 1 Note in Cousera
tags:
  - 机器学习
  - Cousera
  - 林轩田
categories: 学习笔记
mathjax: true
abbrlink: bc80cb40
date: 2019-04-01 14:51:44
copyright: true
---
An introduction for Machine learning

<!-- more -->

# What is machine learning

## What is learning

learning: acquiring skill with experience accumulated from observations

machine learning: acquiring skill with experience accumulated/computed from data

skill: improve some performance measure(e.g.prediction accuracy)

## Why use machine learning

- 'define' trees and hand-program: difficult
- learn from data(observations) and recognize: a 3-year-old can do so
- 'ML-based tree recognition system' can be easier to build than hand-programmed system

ML: an alternative route to build complicated systems

## Some Use Scenarios

- when human cannot program the system manually

  ——navigation on Mars

- when human cannot  'define the solution' easily

  ——speech/visual recognition

- when needing rapid decisions that human cannot do

  ——high-frequency trading

- when needing to be user-oriented in massive sacle

  ——consumer-targeted marketing

## Key Essence of  ML

1\. exists some 'underlying pattern' to be learned

——so 'performance measure' can be improved

2\. but no programmable(easy) definition

——so 'ML' is needed

3\. somehow there is data about the pattern

——so ML has some 'inputs' to learn from

 # Application of ML

# Components of Learning:

## Basic Notations


- input: x $ \in $ *X*

- output: y $ \in $ *Y*

- unknow pattern to be learned $ \Leftrightarrow $ target function:
  *f*: *X* $ \rightarrow $ *Y*

- data $\Leftrightarrow$ training examples: $D =\{(x_1,y_1),(x_2.y_2),...,(x_n,y_n)\}$

- hypothesis $\Leftrightarrow$ skill with hopefully good performance :

  $ g:X \rightarrow Y$

# Machine Learning and Other Fields

## Machine Learning and Data Mining

| Machine Learning                                             | Data Mining                                                |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
| use data to compute hypothesis *g* that approximates target *f* | use **huge** data to find **property that** is interesting |

- if 'interesting property' same as 'hypothesis' that approximate target——ML=DM

- if 'interesting property' related to 'hypothesis ' that approximate target——DM can help ML,and vice versa(often,but not always)

- traditional DM also focuses on efficient computation in large database

## Machine Learning and Artificial Intelligence

| Machine Learning                                             | Artificial Intelligence                               |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| use data to compute hypothesis *g* that approximates target *f* | compute **something that shows intelligent behavior** |

- $ g\approx f$ is something that shows intelligent behavior——ML can realize AI,among other routes
- e.g. chess playing
  - traditional AI: game tree
  - ML for AI :learning from board data

**ML is one possible route to realize AI**

## Machine Learning and Statistics

| Machine Learning                                             | Statistics                                              |
| ------------------------------------------------------------ | ------------------------------------------------------- |
| use data to compute hypothesis *g* that approximates target *f* | use data to **make inference about an unknown process** |

- *g* is an inference outcome,*f* is something unknown——statistics **can be used to achieve ML**
- traditional statistics also focus on **provable results with math assumptions**,and care less about computation

**statistics:many useful tools for ML**

# Summary

1\. What is ML

-use data to approximate target

2\. Application of ML

-alomost everywhere

3\. Components of ML

-$ A$ takes $D$ and $H$ to get $g$

4\. ML and other fields

-related to DM,AI and Stats

 # Appendix

預備知識

作業零 (機率統計、線性代數、微分之基本知識)

參考書籍

Learning from Data: A Short Course , Abu-Mostafa, Magdon-Ismail, Lin, 2013.

經典文獻

F. Rosenblatt. The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6):386-408, 1958. (第二講：Perceptron 的出處)

W. Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963. (第四講：Hoeffding's Inequality)

Y. S. Abu-Mostafa, X. Song , A. Nicholson, M. Magdon-ismail. The bin model, 1995. (第四講：bin model 的出處)

V. Vapnik. The nature of statistical learning theory, 2nd edition, 2000. (第五到八講：VC dimension 與 VC bound 的完整數學推導及延伸)

Y. S. Abu-Mostafa. The Vapnik-Chervonenkis dimension: information versus complexity in learning. Neural Computation, 1(3):312-317, 1989. (第七講：VC Dimension 的概念與重要性)

參考文獻

A. Sadilek, S. Brennan, H. Kautz, and V. Silenzio. nEmesis: Which restaurants should you avoid today? First AAAI Conference on Human Computation and Crowdsourcing, 2013. (第一講：ML 在「食」的應用)

Y. S. Abu-Mostafa. Machines that think for themselves. Scientific American, 289(7):78-81, 2012. (第一講：ML 在「衣」的應用)

A. Tsanas, A. Xifara. Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. Energy and Buildings, 49: 560-567, 2012. (第一講：ML 在「住」的應用)

J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. Introduction to the special issue on machine learning for traffic sign recognition. IEEE Transactions on Intelligent Transportation Systems 13(4): 1481-1483, 2012. (第一講：ML 在「行」的應用)

R. Bell, J. Bennett, Y. Koren, and C. Volinsky. The million dollar programming prize. IEEE Spectrum, 46(5):29-33, 2009. (第一講：Netflix 大賽)

S. I. Gallant. Perceptron-based learning algorithms. IEEE Transactions on Neural Networks, 1(2):179-191, 1990. (第二講：pocket 的出處，注意到實際的 pocket 演算法比我們介紹的要複雜)

R. Xu, D. Wunsch II. Survey of clustering algorithms. IEEE Transactions on Neural Networks 16(3), 645-678, 2005. (第三講：Clustering)

X. Zhu. Semi-supervised learning literature survey. University of Wisconsin Madison, 2008. (第三講：Semi-supervised)

Z. Ghahramani. Unsupervised learning. In Advanced Lectures in Machine Learning (MLSS ’03), pages 72–112, 2004. (第三講：Unsupervised)

L. Kaelbling, M. Littman, A. Moore. reinforcement learning: a survey. Journal of Artificial Intelligence Research, 4: 237-285. (第三講：Reinforcement)

A. Blum. On-Line algorithms in machine learning. Carnegie Mellon University,1998. (第三講：Online)

B. Settles. Active learning literature survey. University of Wisconsin Madison, 2010. (第三講：Active)

D. Wolpert. The lack of a priori distinctions between learning algorithms. Neural Computation, 8(7): 1341-1390. (第四講：No free lunch 的正式版)

T. M. Cover. Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. IEEE Transactions on Electronic Computers, 14(3):326–334, 1965. (第五到六講：Growth Function)

B. Zadrozny, J. Langford, N. Abe. Cost sensitive learning by cost-proportionate example weighting. IEEE International Conference on Data Mining, 2003. (第八講：Weighted Classification)