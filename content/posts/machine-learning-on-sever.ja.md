---
title: サーバーへの自動機械学習のデプロイ
copyright: true
tags:
  - Machine Learning
  - Linux
  - Server
categories: 学習ノート
abbrlink: c23e0e6e
slug: deploying-automl-on-server
date: 2019-06-05 14:25:11
lang: ja
mathjax:
---

卒業インターンシップの機会を利用して、コードを研究室のサーバーにデプロイしました。ここでその全過程を記録します。

<!-- more -->

# 研究室サーバーへのログインと基本操作

サーバーホストに直接ログインする以外に、接続ソフトウェアを使用してリモートログインすることもできます。ここではPuTTYを使用しています。

PuTTYはオープンソースの接続ソフトウェアで、主にSimon Tathamによってメンテナンスされ、MITライセンスの下で提供されています。含まれるコンポーネント：PuTTY、PuTTYgen、PSFTP、PuTTYtel、Plink、PSCP、Pageant。デフォルトのログインプロトコルはSSHで、デフォルトポートは22です。PuTTYはSSH、Telnet、Serialなどのプロトコルをサポートするリモートサーバー接続に使用されます。SSHが最も一般的に使用されます。リモートLinux管理に非常に便利で、主な利点は以下の通りです：

- 完全に無料でオープンソース
- Windowsを完全サポート
- SSH1とSSH2を完全サポート
- ポータブルソフトウェア、インストール不要、ダウンロード後にデスクトップショートカットを作成するだけ
- 非常に小さいサイズ、1MB未満
- シンプルな操作、すべての操作が1つのコントロールパネルで完結

PuTTYのダウンロードページ：<https://putty.org/>、適切なバージョンを選択してダウンロード・インストールしてください。

ダウンロード後、Saved SessionsにサーバーIPアドレスを入力し、Saveをクリック、次にOpenをクリックしてセッションを開始します。ユーザー名とパスワードを入力してログインします。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20190605143842.png)

次に、基本コマンドを使用してサーバーのシステムバージョンとPythonバージョンを確認します。

入力：

```bash
cat /etc/redhat-release
```

返されるシステムバージョンはCentOS 7.3.1611 Redhatです。

入力：

```bash
python
```

返されるPythonバージョンは2.7です。

mkdirコマンドを使用して新しいディレクトリを作成します。以降の操作はすべてこのディレクトリで行います。

```bash
mkdir amf
```

CPUE自動予測システムの環境設定

研究室のLinuxサーバーバージョンはCentOS 7.3.1611 Redhatで、デフォルトのPythonバージョンは2.7です。そのため、まずPython 3をサーバーにインストールする必要があります。

# Python 3のインストール

## RPMパッケージインストール

今回のインストールはIUSコミュニティのRPMパッケージを使用します。IUSは「Inline with Upstream Stable」の略で、新しいバージョンのRPMパッケージを提供するコミュニティです。詳細は[公式ドキュメント](https://ius.io/GettingStarted/#install-via-automation)を参照してください。

使用する具体的な操作は以下の通りです（一部の操作にはsudo権限が必要です）：

(1) IUSリポジトリを追加：

```bash
yum -y install https://centos7.iuscommunity.org/ius-release.rpm
```

(2) キャッシュメタデータを作成：

```bash
yum makecache
```

(3) Python 3.6をインストール：

```bash
yum install python36u

yum -y install python36u-pip

yum -y install python36u-devel
```

(4) 環境テスト：python3.6と入力して以下のテキストが表示されれば、インストール成功です。

```python
Python 3.6.8 (default, May  2 2019, 20:40:44)

[GCC 4.8.5 20150623 (Red Hat 4.8.5-36)] on linux

Type "help", "copyright", "credits" or "license" for more information.
```

## 仮想環境の設定

システムに複数のPythonバージョンが存在するため、環境汚染を避けるためにvirtualenvを使用して独立した仮想環境を作成しました。手順は以下の通りです：

```bash
python3.6 -m venv py3

source py3/bin/activate
```

実行後、コマンドラインの前に(py3)が表示され、仮想環境に入ったことを示します。python -Vと入力してpython3.6.8が返されれば、インストール成功です。

## 機械学習環境の設定

自動予測システムはauto-sklearnをベースに開発されており、scipy、scikit-learn、pandas、numpyなどのライブラリのサポートが必要です。前のステップでpipをインストールしました。

注意：設定前に、前のステップで設定した仮想環境に入る必要があります。

以下のコマンドを順番に実行してインストールします：

```bash
pip install pandas

pip install scipy

pip install scikit-learn

pip install auto-sklearn

pip install matplotlib

pip install xlrd

pip install openpyxl
```

auto-sklearnのインストール中に以下のエラーが発生しました：

```bash
Error: Syntax error in input(3).     error: command 'swig' failed with exit status 1
```

プロンプトによると、このエラーはswigが原因です。SWIGは本質的にコードジェネレーターで、C/C++プログラムから他の言語へのラッパーコードを生成します。

swigのバージョンは2.0.10で、Python 2.7に対応するバージョンでした。そのため、swigをバージョン3にアップグレードする必要があります。

まずswig2をアンインストール：

```bash
yum -y remove swig
```

次にswig 3をインストール：

```bash
yum install swig3
```

インストール後、実行：

```bash
swig -version
```

swigのバージョンが3.0.12に変わったことが確認できます。

pip install auto-sklearnを再実行してインストールを完了します。

インストール完了後、仮想環境でpythonと入力し、Pythonインタープリターに入り、以下のコマンドを実行します。エラーがなければ、環境設定は完了です。Ctrl+Dを押して仮想環境とPuTTYを終了します。

```python
import autosklearn.regression

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np
```

# サーバーとWindows間のファイル転送

プログラムを実行する前に、データをサーバーに転送する必要があります。ここではPuTTYが提供するPSFTPツールを使用します。PuTTYと一緒にインストール済みです。

PSFTPを開き、「open サーバーアドレス」と入力してログインを完了します。putコマンドでローカルファイルをサーバーに転送し、getコマンドでサーバーからファイルを取得します。

# プログラムのデバッグ

PSFTPを使用してソースコードをサーバーに転送し、仮想環境に入り、コマンドラインで入力：

```bash
python code.py
```

プログラムが実行を開始するのが確認できます。

プログラムがエラーを報告した場合、システム内蔵のvimエディタを使用して編集とデバッグを行う必要があります。vimには3つのモードがあります：コマンドモード、入力モード、ボトムラインモード。

(1) コマンドモード：

vimを起動すると、コマンドモードに入ります。この状態では、キーボード操作は文字入力ではなくコマンドとして認識されます。

よく使うコマンド：
- i：入力モードに切り替え
- x：カーソル位置の文字を削除
- :：ボトムラインモードに切り替え

(2) 入力モード：

コマンドモードでiを押すと入力モードに入ります。入力モードでは以下のキーが使用できます：
- 文字キーとShiftの組み合わせで文字入力
- ENTERで改行
- BACKSPACEでカーソル前の文字を削除
- DELでカーソル後の文字を削除
- 矢印キーでカーソル移動
- HOME/ENDで行頭/行末に移動
- Page Up/Page Downでページスクロール
- Insertで入力/置換モードを切り替え
- ESCで入力モードを終了しコマンドモードに戻る

(3) ボトムラインモード：

コマンドモードで:（コロン）を押すとボトムラインモードに入ります。基本コマンド（コロン省略）：
- q：プログラム終了
- w：ファイル保存
- ESCでいつでもボトムラインモードを終了

vimでtest.txtというファイルを編集または作成するには、仮想環境で入力：

```bash
vim test.txt
```

これでノーマルモードに入ります。iを押すと入力モードに入り、ステータスバーに--INSERT--が表示されます。編集後、ESCを押してノーマルモードに戻ります。ノーマルモードで:を押し、wqと入力して保存して終了します。

他にも多くの高度なコマンドがありますが、ここでは省略します。
