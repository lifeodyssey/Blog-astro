---
title: Java
tags:
  - Java
  - 'Software Engineering'
categories: 学習ノート
abbrlink: 7f1ae6d2
date: 2022-04-11 17:49:43
mathjax:
copyright:
password: GTB2o22
lang: ja
---

なぜGoを使わないのか？後でKotlinも学ばないといけないのに。

<!-- more -->

# インストールと環境設定

```powershell
winget install Oracle.JDK.17 --location [インストール先]
```

Windowsコマンドラインオプション。

## 環境変数の設定

- JDKインストール後、JAVA_HOME環境変数を設定
- `java -version`で確認

# Java基礎

ずっと避けてきたオブジェクト指向がまた来た。

## 継承

- `extends`はクラスを継承
- `implements`はインターフェースを実装

## オーバーライド vs オーバーロード

オーバーライド：子クラスが親のメソッドを書き換え
オーバーロード：同じメソッド名、異なるパラメータ

# クロスオリジン問題

ブラウザの同一オリジンポリシーによる。同一オリジン = プロトコル + ドメイン + ポート
