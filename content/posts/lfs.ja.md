---
title: lfs
tags:
  - python
  - git
categories: 学習ノート
mathjax: true
abbrlink: 442707f1
date: 2021-11-29 12:06:20
copyright:
password:
lang: ja
---

GitHubの大きなファイル

<!-- more -->

最近作業中によくこのエラーに遭遇します：

fatal: sha1 file '<stdout>' write error: Broken pipe

実は原因はGitHubが最大100MBまでしかサポートしていないからです。普段のコードには十分ですが、HDFやNCファイルがあると無理です。

https://git-lfs.github.com/ からLFSをインストールした後、コマンドライン（PyCharmは使えないようです）で操作できます。

実は一番良い方法はデータファイルを別々に保存することですが、途中で一度commitした時にファイルも一緒に追加してしまい、パスを変更しても一緒にpushされてしまいます。

仕方なく、まずアップロードしてからパスを移動し、その後commitしてpushするしかありませんでした。

また、LFSにも制限があり、1GBまでしかアップロードできず、それ以上は有料です。本当に面倒です。

今のところ、より良い方法は見つかっていません。

## [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)

より良い方法がちょうど見つかりました。

[このページ](https://stackoverflow.com/questions/2100907/how-to-remove-delete-a-large-file-from-commit-history-in-the-git-repository)で見つけました。

すでにGitHubにファイルをアップロードしてしまった場合は、BFGをダウンロードできます。

そして：

```bash
$ git clone --mirror git://example.com/some-big-repo.git
$ java -jar bfg.jar --strip-blobs-bigger-than 100M my-repo.git
$ git gc --prune=now --aggressive
```

完了です。

この方法にはJREが必要なことに注意してください。

## git-filter-branch

しかし上記の方法は私にとっては無限ループです。

ファイルが大きすぎてGitHubにpushできないからです。

だからそのmirrorをcloneしてcommitをクリーンにする方法が使えません。

別の方法は：

```bash
$ git filter-branch -f --index-filter "git rm -rf --cached --ignore-unmatch *.HDF" -- --all
```

私の場合、大きなファイルは全部HDFだとわかっているので、ワイルドカードを使えばいいです。

そして：

```bash
$ rm -rf .git/refs/original/
$ git reflog expire --expire=now --all
$ git gc --prune=now
$ git gc --aggressive --prune=now
```

最後に全部pushします：

```bash
$ git push --all --force
```
