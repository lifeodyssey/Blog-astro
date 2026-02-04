---
title: サーバー上の海色環境構築
tags:
  - Ocean Color
  - Linux
categories: 学習ノート
abbrlink: eff06e8d
date: 2021-12-06 15:16:53
mathjax:
copyright:
password:
lang: ja
---

Python, SeaDAS&OCSSW, SNAP

<!-- more -->

# Python

最初、うっかりbaseにいくつかインストールしてしまいましたが、削除できます。できるだけpysoc環境で実行してください。

```bash
$ mkdir -p miniconda
$ cd miniconda
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda.... .sh
$ conda create --name pysoc python=3.7
$ conda activate pysoc
$ conda install -c conda-forge py6s
$ conda install seaborn
$ conda install -c conda-forge xarray dask netCDF4 bottleneck
$ conda install -c conda-forge cartopy
$ conda install -c conda-forge pyresample
$ conda install -c anaconda basemap
$ conda install -c conda-forge basemap-data-hires
$conda install opencv-python
```

インストール後、py6sをテストする必要があります

```python
$ python
>>> from Py6S import *
>>> SixS.test()
6S wrapper script by Robin Wilson
Using 6S located at <PATH_TO_SIXS_EXE>
Running 6S using a set of test parameters
The results are:
Expected result: 619.158000
Actual result: 619.158000
#### Results agree, Py6S is working correctly
```

このように表示されれば正しいです。

## Jupyter

プログラムを書くのに便利なようにjupyterをインストールしました

```bash
$ pip install jupyterlab
$ conda install -c conda-forge nodejs
$ conda install -c conda-forge python-lsp-server r-languageserver
$ jupyter labextension install @jupyterlab/toc
$ conda install -c conda-forge jupyterlab-drawio
$ pip install jupyterlab_latex
$ pip install lckr-jupyterlab-variableinspector
$ conda install -c conda-forge jupyterlab_execute_time
$ conda install -c conda-forge nb_conda_kernels
$ jupyter lab build
$ jupyter labextension enable all
```

これはリモートサーバー上なので、設定が必要です

```bash
$ jupyter lab --port=8888 --no-browser --allow-root
```

自分のコンピュータで

```bash
$ ssh -N -f -L 8880:localhost:8888 root@123435
```

最後に、サーバーに表示されたURLをブラウザに追加するだけです。

プログラマーにファイアウォールを開けてもらうことを忘れずに。

# SeaDAS

## 基本環境

```bash
$ sudo yum -y install gcc
$ sudo yum -y install git
$ sudo yum -y install curl
$ sudo yum -y install wget
```

RedHatなので、apt-getがありません。

```bash
$ sudo yum -y-repository ppa:webupd8team/java
$ sudo yum -y update
$ sudo yum -y install oracle-java8-installer
$ sudo yum -y install oracle-java8-set-default
$ java -version
```

表示

```bash
java version "1.8.0_311"
Java(TM) SE Runtime Environment (build 1.8.0_311-b11)
Java HotSpot(TM) 64-Bit Server VM (build 25.311-b11, mixed mode)
```

```bash
$ wget https://oceandata.sci.gsfc.nasa.gov/SeaDAS/installer/7.3.1/seadas_7.3_linux64_installer.sh

$ bash seadas...sh

```

これはGUIインストールが不要な最後のバージョンです。どうりで多くの論文が7.3を使い続けているわけです。

Where should SeaDAS be installed?
[/opt/seadas-7.3]

>不思議なこと
>
>Linuxのパスがこのように動作することを初めて発見しました
>
>cd / でルートディレクトリに入ります
>
>このディレクトリはすべてのユーザーで共有されています
>
>一方、cd ~ で自分のディレクトリに入ります
>
>勉強になりました

## OCSSW

最も重要な部分

```bash
$　wget https://oceandata.sci.gsfc.nasa.gov/ocssw/install_ocssw.py
$ chmod +x install_ocssw.py
$ ./install_ocssw.py --install-dir=/root/SeaDAS/seadas-7.3/ocssw --aqua --seawifs --goci
$ mv /opt/seadas-7.3 ~/SeaDAS
$ vim ~ .bashrc
```

入力

```bash
export SEADASPATH=/root/SeaDAS/seadas-7.3
export PATH=$PATH:$SEADASPATH/bin
```

where: `[SeaDAS_install_dir]` はSeaDASをインストールしたディレクトリです。

完了したら保存します。

ここで、OCSWWパスにインストールしたファイルが見つからないことに気づきました。調べてみると、~ではなく/にインストールされていました。上のダウンロード場所でパスを変更できますが、変更するのを忘れていました。

移動し直すしかありませんでした。

テストコマンド

```bash
$ source .../OCSSW_bash.env
$ which l2gen
```

表示

```bash
/root/SeaDAS/seadas-7.3/ocssw/bin/l2gen
```

これで完了です。

その後、より完全な設定をbashrcに追加しました

```bash
export SEADASPATH=~/SeaDAS/seadas-7.3
export PATH=$PATH:$SEADASPATH/bin
export PYTHON_PROGRAMS=/root/miniconda3/bin/python
export PYTHONPATH=$PYTHONPATH:$PYTHON_PROGRAMS:$PYTHON_PROGRAMS/utilities
export LOCAL_RESOURCES=$PYTHON_PROGRAMS/local_procesing_resources
export OCSSWROOT=$SEADASPATH/ocssw
export OCSSW_DEBUG=0 # set to 1 for debugging
source $OCSSWROOT/OCSSW_bash.env
```

## ファイル転送

以前はPuttyやFileZillaを直接使っていましたが、会社のプログラマーにSCPを使うように言われました

```bash
$ scp SourceFile user@server:directory/TargetFile
```

戻す場合は

```bash
$ scp user@server:directory/SourceFile TargetFile
```

## 大気補正

NASAのデータ形式の更新により、以前のスクリプトとは少し異なります

```bash
#Example script to process L1A files up to L2, put in same directory as
# your L1A files
# Run script by typing on Terminal Command Line:
# bash l1a_to_l2.sh input.txt
# input.txt contains list of files [ create by: ls -1 *L1A_LAC* > input.txt ]

LIST=$1
if [ -z "$LIST" ]
  then
    echo "No input file list supplied"
    exit 1
fi

if [[ ! -a $LIST ]]; then
    echo "$LIST does not exist!"
    exit 1
fi

mkdir -p done
mkdir -p l2_lac

for FILE in $(cat $LIST);
do
    echo "Working on $FILE"
    echo ${FILE}
    # get file basename (no file extension)
    BASE=`basename $FILE .L1A_LAC`
    echo ${BASE}
    GEOFILE=${BASE}.GEO
    echo ${GEOFILE}
    L1BFILE=${BASE}.L1B_LAC
    echo ${L1BFILE}
    L2FILE=${BASE}.L2_LAC.x.hdf
    echo ${L2FILE}
    ancfile=${BASE}.L1A_LAC.x.hdf.anc
    echo "Creating GEOFILE $GEOFILE"
    modis_GEO.py -v $FILE -o $GEOFILE
    echo "Creating L1B file"
    modis_L1B.py -v $FILE $GEOFILE
    echo "Creating anc file"
    getanc.py -v $FILE
    echo "Create L2 file"
    l2gen ifile=$L1BFILE geofile=$GEOFILE ofile=$L2FILE par=$ancfile l2prod="Kd_490 Rrs_nnn angstrom aot_869 chlor_a ipar nLw_nnn nflh par pic poc rhos_nnn" aer_opt=-2
    echo "Cleaning up"
    rm -v ${BASE}.L1B* $GEOFILE $ancfile
    #mv -v $FILE done/
    mv -v $L2FILE l2_lac/
done
```

### うまくいかなかった

テストデータを実行した後、動作しないことがわかりました。

再インストールするしかありませんでした。

そしてこのネットワークも不安定でした。

![image-20211209155834683](C:/Users/zhenjia/AppData/Roaming/Typora/typora-user-images/image-20211209155834683.png)

典型的な404

調べてみると、公式サイトがメンテナンス中でした。

## うまくいった

```bash
$ bzip2 -d *.bz2
$ ls -1 *L1A_LAC* > input.txt
$ bash l1_to_l2.sh inupt.txt
```
