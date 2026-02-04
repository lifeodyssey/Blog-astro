---
title: WSL+anaconda+jupyter
tags:
  - 機械学習
  - Linux
categories: 学習ノート
copyright: true
abbrlink: 60b81fb7
date: 2021-12-04 19:00:45
mathjax:
password:
lang: ja
---

以前autosklearnをよく使っていましたが、Windowsでは動作しません。以前のようにUbuntuをインストールしようと思いましたが、面倒に感じたのでWSLを試してみました。ここに設定の全過程を記録します。

<!-- more -->

# WSL

管理者権限でcmdを開き、以下を入力します：

```bash
$ wsl --install
```

再起動後、MS StoreからUbuntuをダウンロードします。私はUbuntu 20.04 LTSをダウンロードしました。

その後、コマンドラインでwslと入力するだけで開始できます。

![image-20211204211035801](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewinimage-20211204211035801.png)

これはデフォルトでCドライブにインストールされることに注意してください。

[Ubuntuの一般的なコマンド](https://www.cnblogs.com/linuxws/p/9307187.html)の参考リンクです。

# anaconda

ここではminicondaをインストールしました。そんなに多くのパッケージは必要ないと感じたからです。

コマンドは以下の通りです：

```bash
$ mkdir -p miniconda
$ cd miniconda
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

そしてこのステップで詰まってしまい、長い間先に進めませんでした。

清華ミラーからminicondaをダウンロードするしかありませんでした。py37バージョンをダウンロードし、minicondaフォルダに直接コピーします。ファイルディレクトリは通常c/users/username/minicondaです。

その後：

```bash
$ bash Miniconda.... .sh
$ conda create --name pysoc python=3.7
$ conda activate pysoc
```

そして中国のユーザーには欠かせないステップ、清華ミラーへの切り替えです。

## 挿話

その前に、ちょっとした挿話がありました。

rebootで直接再起動しようとしましたが、うまくいかず、以下のように表示されました：

```bash
$ System has not been booted with systemd as init
$ system (PID 1). Can't operate.
$ Failed to connect to bus: Host is down
$ Failed to talk to init daemon.
```

たくさん調べましたが、問題が何なのか正確には理解できませんでしたが、最終的にこのように解決しました：

```bash
$ sudo -s
$ wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
$ sudo dpkg -i packages-microsoft-prod.deb
$ sudo apt-get update
$ apt search dotnet-sdk
```

これはdotnet-sdkを直接インストールする際に発生する「Unable to locate package dotnet-sdk」エラーを解決するためです。

続けて：

```bash
$ sudo apt install -y daemonize dbus dotnet-runtime-5.0 gawk libc6 libstdc++6 policykit-1 systemd systemd-container

```

必要なモジュールをインストールします：

```bash
$ sudo apt install apt-transport-https

$ sudo wget -O /etc/apt/trusted.gpg.d/wsl-transdebian.gpg https://arkane-systems.github.io/wsl-transdebian/apt/wsl-transdebian.gpg

$ sudo chmod a+r /etc/apt/trusted.gpg.d/wsl-transdebian.gpg

$ sudo cat << EOF > /etc/apt/sources.list.d/wsl-transdebian.list
deb https://arkane-systems.github.io/wsl-transdebian/apt/ $(lsb_release -cs) main
deb-src https://arkane-systems.github.io/wsl-transdebian/apt/ $(lsb_release -cs) main
EOF

$ sudo apt update
```

wsl-transdebianのリポジトリを設定します。

```bash
$ sudo apt install -y systemd-genie
$ genie -i
$ genie -s
$ genie -l
```

genieをインストールして使用します。

そして見事にクラッシュしました。

調べた結果、理由は：

>So why don't normal `reboot`/`shutdown` commands work? Two reasons. First, as covered by Bengt, WSL doesn't currently support systemd, and Ubuntu simply links these two legacy commands to `/usr/bin/systemctl` (the systemd control utility). You can see this with `ls -l /usr/sbin`.
>
>But even if these were the legacy commands which directly called [Linux's shutdown API](https://stackoverflow.com/q/28812514/11810933), it wouldn't work. Microsoft doesn't typically hook up API's that interact directly with the hardware, instead providing virtualization interfaces where necessary. But in the case of starting and stopping a WSL instance, it's just so "lightweight" (as discussed above) that there's not any real reason to do so.

結局、別のcmdを開くしかありませんでした：

```bash
$ wsl --list
$ wsl --terminate Ubuntu
#or wsl shutdown to shut all the sub system
```

再起動して入った後、前に「base」が表示されていればconda環境に入っています。

## ソースの変更

```bash
$　conda config --add channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/'
$ conda config --set show_channel_urls yes
```

## 環境の設定

```bash
$ conda create --name pysoc python=3.7
$ conda activate pysoc
```

絶対に、絶対に、絶対にbase環境で何かを実行したりインストールしたりしないでください！

定番のパッケージインストール時間：

```bash
$ conda install gxx_linux-64 gcc_linux-64 swig
$ conda config --add channels conda-forge
$ conda config --set channel_priority strict
$ conda install auto-sklearn
$ conda install -c conda-forge geopandas
```

そしてまた見事にクラッシュしました。

![image-20211205111723183](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewinimage-20211205111723183.png)

環境を削除して再作成してみましたが、うまくいきませんでした。仕方なく、システムを再インストールすることにしました。

Ubuntuの設定を開いてresetをクリックするだけです。

![image-20211205114045880](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewinimage-20211205114045880.png)

再インストール後、もう一度やり直しますが、今回はreboot関連のgenieパッケージはインストールしませんでした。

今回はクラッシュしませんでした。上記の2つのパッケージをインストールした後、続けて：

```bash
$ conda install seaborn
$ conda install -c conda-forge xarray dask netCDF4 bottleneck
$ conda install -c conda-forge cartopy
$ conda install -c conda-forge pyresample
$ conda install python-ternary
```

よく使うのはこれくらいだと思います。何か足りなければ後で試してみます。

# Jupyter

```bash
$ pip install jupyterlab
$ conda install -c conda-forge nodejs
$ conda install -c conda-forge python-lsp-server r-languageserver
$ jupyter labextension install @jupyterlab/toc
$ conda install -c conda-forge jupyterlab-drawio
$ pip install jupyterlab_latex
$ pip install lckr-jupyterlab-variableinspector
$ conda install -c conda-forge jupyterlab_execute_time
$ conda install ipykernel
$ jupyter lab build
$ jupyter labextension enable all
$ python -m ipykernel install --name acolite

```

以前書いた[この記事](https://lifeodyssey.github.io/posts/c958af43.html)を参考にできます。

その後、jupyter labと入力するだけで起動できます。

大量の情報が一気に流れて、cmdで次のページに飛んでしまい、tokenが見えなくなることがあります。ctrl+cを押すとtokenが表示されるので、それをブラウザにコピーしてください。

git cloneしたリポジトリもCドライブにあるので、以前書いたjupyterlabをそのまま開いて使えます。

全部で7〜8時間かかりました。そのほとんどはrebootを設定しようとしたためです。それがなければ約4時間で終わっていたでしょう。

アクセスパスが/mnt/c(defgh...)に変わることに注意してください。

## またクラッシュ

jupyterでautosklearnをインポートしようとしたら、autosklearnがないと表示されました。condaコマンドでインストールされていなかったようです。

再度インストールしてみたところ、geopandasとautosklearnに競合があることがわかりました。呆れました。

最終的に：

```bash
$ pip3 install auto-sklearn
```

で解決しました。

また、autosklearnの一部のキーワードは現在（2021.12.5）、学部時代に使っていたものと異なっていることに注意してください。

# CPUとメモリの制限

1日実行した後、まだ結果が出ていないことに気づきました。更新すると「kernel seems to die」と表示されました。

調べたところ、割り当てられたメモリが不足していたためでした。

```bash
$ wsl --shutdown
$ notepad "$env:USERPROFILE/.wslconfig"
```

以下を書き込みます：

[wsl2]
memory=3GB   # Limits VM memory in WSL 2 up to 3GB
processors=4 # Makes the WSL 2 VM use two virtual processors

好きな量を書いてください。

wslで`free -h`を使用してメモリ割り当て状況を確認できます。

# バグ

最近、外付けハードドライブが認識されない問題に遭遇しました。

「No such device」と表示され続けました。

そして[これ](https://dowww.spencerwoo.com/)を見つけました。

本当に宝物です！今すぐブックマークしてください。

ただし、この問題は私が遭遇したものとは異なります。

## パスワードリセット

https://zikin.org/wsl-forgot-passwd/

```
https://raw.githubusercontent.com/acolite/acolite_luts/main/ACOLITE-LUT-202110-Reverse/L8_OLI/ACOLITE-LUT-202110-MOD2-reverse-L8_OLI-romix-8.nc
```

```bash
win x A
powershell Admin
wsl -u root

passwd root

passwd zikin
```
