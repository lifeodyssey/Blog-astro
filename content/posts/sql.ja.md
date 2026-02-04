---
title: SQL
tags:
  - Java
  - 'Software Engineering'
categories: 学習ノート
abbrlink: dbd66adf
password: woshiyelanxiaojiedeggou
date: 2022-06-04 18:34:45
mathjax:
copyright:
lang: ja
---

ずっと後回しにしていたものがついに来ました。

<!-- more -->

# SQLクエリ

複数テーブルからクエリする時は、必ずどのテーブルか明記する。

MySQLはTOPをサポートしていない、代わりにLIMITを使う。

## Join

Inner Join: 両方のテーブルにマッチするキーがある
Left Join: Aを全て保持、Bがあれば結合、なければnull

# JDBC

## Connection

各メソッドでConnectionを作るのが面倒だと思い、全メソッドで1つのconnectionを使おうとしました。結果：

java.sql.SQLNonTransientConnectionException: No operations allowed after connection closed.

try-with-resourcesはtryブロック後に自動でリソースを解放するため、最初のtry-with-resources後にconnectionが解放され、後続の呼び出しが失敗しました。
