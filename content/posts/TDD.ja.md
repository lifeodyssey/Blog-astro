---
title: TDD
tags: 'Software Engineering'
categories: 学習ノート
password: yuanlainiyewanyuanshen
abbrlink: 715050f6
date: 2022-07-18 16:22:18
mathjax:
copyright:
lang: ja
---

# TDDデモ

# タスキング

## タスキング理論

### タスキング鉄の三角形

- 価値がある
  - ビジネス価値がある
  - 機能を実装する

- 十分に小さい
  - 作業者が作業を開始できる
  - 「動けない」や「盲目的に作業」ではない

- 人間の言葉で話す
  - コミュニケーション：3日後でも理解できる

# クリーンコード

## 1-10-50ルール

- メソッドごとに最大1レベルのインデント
- メソッドごとに最大10行
- クラスごとに最大50行

## SOLID原則

- 単一責任
- 開放閉鎖
- リスコフ置換
- インターフェース分離
- 依存性逆転

# TDD

## TDDの三法則

1. 失敗するテストを先に書く
2. パスする最小限のコードを書く
3. リファクタリング

## テスト構造：Given-When-Then (AAA)

- Arrange (Given)
- Act (When)
- Assert (Then)

## 統合テスト vs 単体テスト

統合テスト：実際のファイル/DBでテスト
単体テスト：単一セクションのみテスト
