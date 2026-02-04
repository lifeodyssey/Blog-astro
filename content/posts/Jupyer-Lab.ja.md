---
title: Jupyter Lab
tags:
  - python
categories: 学習ノート
mathjax: true
abbrlink: c958af43
date: 2021-05-12 11:25:46
copyright:
lang: ja
---

今のところJupyterでしかコードを書けませんが、元のnotebookは数MBのファイルを書くたびにどんどん大きく重くなってきたので、Labに切り替えました。

使う際に設定したことをいくつか紹介します。

<!-- more -->

# Extension

（最初から困惑 - 実は[日本語のブログ](https://qiita.com/canonrock16/items/d166c93087a4aafd2db4)を参考にしています、中国語のリソースが見つからなかった）

Extensionをインストールするには、NodeJSが必要です（なぜ？）

```bash
conda install -c conda-forge nodejs
```

Listでインストール済みのものを確認できます：

```
jupyter labextension list
```

主に[Awesome Jupyter](https://github.com/markusschanta/awesome-jupyter)からいくつかインストールしました。基本的に元のNotebookと同じようにするため、lsp、toc、drawio、latex、variableinspectorなど。リストにないものとして[jupyterlab-execute-time](https://github.com/deshaw/jupyterlab-execute-time)と[jupyterlab-kite](https://github.com/kiteco/jupyterlab-kite)があります。

最後にこれを実行：

```bash
jupyter lab build
jupyter labextension enable all
```

またはLabの設定で手動でextensionを有効にします。

不便な点は、Variable inspectorがフロートできないことです。

もう一つ - 一般ユーザーとして、kiteは必要ないと感じます。lspで十分です。

とはいえ、やっぱりnotebookの方が使いやすいと感じます。
