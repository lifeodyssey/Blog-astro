---
title: Tree Family in Machine learning and XGBoost
tags:
  - 机器学习
  - XGBoost
  - Auto-sklearn
categories: 学习笔记
abbrlink: 760defdb
date: 2021-12-23 10:33:20
mathjax:
copyright:
password:
---

因为研究的原因又回到了老本行，要用auto-sklearn和XGBoost，这次一定要把我自己讲明白



一些地址

https://towardsdatascience.com/beyond-grid-search-hypercharge-hyperparameter-tuning-for-xgboost-7c78f7a2929d

https://issueexplorer.com/issue/automl/auto-sklearn/1297

https://www.kaggle.com/residentmario/automated-feature-selection-with-sklearn

https://stackoverflow.com/questions/54035645/features-and-feature-importance-in-auto-sklearn-with-one-hot-encoded-features

https://github.com/automl/auto-sklearn/issues/524

https://towardsdatascience.com/feature-preprocessor-in-automated-machine-learning-c3af6f22f015

https://scikit-learn.org/stable/modules/feature_selection.html

https://github.com/automl/auto-sklearn/issues/524

https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_inspect_predictions.html

POC baiyan那个

# What is tree

# Tree or neutral network

# Bagging and Boost(ensemble)



# Hyperparameter selection: the Bayes

这里要顺便补充一下之前写了一半的那个

# How auto-sklearn works and how ‘auto’ it is

# Hands on and integration with original model





# XGBoost

## Bagging and Boosting

共同点：都是由弱分类器构成。弱分类器就是表现不好的分类器

Bagging，训练多个，然后投票加权，弱分类器通常是过拟合的

Boosting，训练多个，然后投票加权，弱分类器通常是欠拟合的

## 提升树

提升树是由基于残差的训练

不由一个模型完成最终预测，将一个模型训练之后的残差作为下一个模型预测的目标值，记为y2，如此循环

最终预测=模型1的预测+模型2的预测+模型3的预测+...

没有投票，每个模型都参与的预测的记过里面

## 模型原理及优化

