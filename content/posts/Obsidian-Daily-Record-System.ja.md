---
title: Obsidian Daily Record System
tags:
  - 雑記
categories: 実用ツール
abbrlink: f36cc8b7
date: 2022-07-23 23:36:22
mathjax:
copyright:
lang: ja
---

この記事は https://lifeodyssey.github.io/posts/b58cbef5.html の続編です。
まず、現在のソリューションの欠点について書きます：
- 日常記録しかできず、Personal Knowledge Management (PKM) には対応していない
- Obsidianの有料サービスを購入していないため、gitに依存して同期しており、全プラットフォームでの同期ができない。デスクトップでしか使用できず、モバイルはまだメモアプリに依存している
- タスクリストのサポートが弱く、この部分はまだMicrosoft ToDoで解決している

将来的にはNextcloudと組み合わせるかもしれません。
主にこれを参考に完成しました：https://diygod.me/obsidian/

<!-- more -->

# 解決すべき問題

元の作者と比べて、私はObsidianに多くのものを追加するのが好きではありません。運動とフィットネスはGoogle Fit、睡眠はAndroid Sleep Companion、お金は家計簿アプリを使っています。

ここでは主にObsidianで以下のことを行います：

1. 毎日のCompleted List、自分を褒めるために10個のことを書く
2. 日常生活で思いつくかもしれない小さな日記。大きな日記はOneNoteと紙の日記に書いており、これはメモ帳のようなもの
3. Kanban、Trelloを使いたくないから

なので、元のバージョンをかなり簡略化しました。

主な要件は以下の通りです：

1. 毎日Calendarをクリックして自動的に日記を生成し、/Year/Month/Weekの順序で配置する
2. 毎週ガントチャート付きの計画表とトグルサマリーを生成する
3. 毎月自動的にkanbanを生成する
4. 年間の大きなガントチャート付き計画表を生成し、年末にレビューするリマインダーを設定する

一つずつ見ていきましょう。

# 具体的な変更

これはPeriodic Noteとtemplaterのfunctionを使って実現できます。私はよくVPNを使うので、IPベースで現在地を取得する方法は使わず、固定の場所を使いました。getWeatherを例にすると：

```bash
curl wttr.in/"$(curl -s --header "user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36" https://api.ip.sb/geoip | /opt/homebrew/bin/jq -r ".city" | sed 's/ /%20/')"\?format="%l+%c%t"
```

これを以下のように変更しました：

```bash

curl wttr.in/City:Province:Country?format="%l+%c%t"
```

その他は変更なしです。
