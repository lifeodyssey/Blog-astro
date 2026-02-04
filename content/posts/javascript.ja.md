---
title: JavaScript
tags:
  - Javascript
  - 'Software Engineering'
categories: 学習ノート
abbrlink: 84ea90f7
password: GTB2o22
date: 2022-06-27 15:45:10
mathjax:
copyright:
lang: ja
---

後でTypeScriptも学ばないと。

<!-- more -->

# 基礎

JSは自動型変換を行う。3/4=0.75

`==`は比較時に型変換を行う。
`===`（厳密等価）は型変換を行わない。

`var`は関数/グローバルスコープ、`let`と`const`はブロックスコープ。

## 配列メソッド

`map()`は各要素に関数を適用した新しい配列を作成。
`reduce()`は各要素にreducer関数を実行し、単一の値に集約。

## thisについて

`this`は動的に親コンテキストにバインドされる。オブジェクトから抽出するとundefinedになる。
