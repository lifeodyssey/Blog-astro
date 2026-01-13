---
title: Ocean color Environment on Server
tags:
  - Ocean Color
  - Linux
categories: 学习笔记
abbrlink: eff06e8d
date: 2021-12-06 15:16:53
mathjax:
copyright:
password:
---

Python, SeaDAS&OCSSW, SNAP

<!-- more -->

# Python

刚开始不小心在base上装了一些，可以删，尽量在pysoc里跑。



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

装好之后有必要测试一下py6s

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

这样的话就是正确的

## Jupyter

为了方便我自己写程序装了jupyter

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

因为这个是在远程服务器 需要一定的配置

```bash
$ jupyter lab --port=8888 --no-browser --allow-root
```

在自己电脑上

```bash
$ ssh -N -f -L 8880:localhost:8888 root@123435
```

最后在浏览器里把服务器那里显示的网址加上就行了

记得让程序员大哥开防火墙

# SeaDAS

## 基础环境

```bash
$ sudo yum -y install gcc
$ sudo yum -y install git
$ sudo yum -y install curl
$ sudo yum -y install wget
```

因为是redhat， 没有apt-get。

```bash
$ sudo yum -y-repository ppa:webupd8team/java
$ sudo yum -y update
$ sudo yum -y install oracle-java8-installer
$ sudo yum -y install oracle-java8-set-default
$ java -version
```

显示

```bash
java version "1.8.0_311"
Java(TM) SE Runtime Environment (build 1.8.0_311-b11)
Java HotSpot(TM) 64-Bit Server VM (build 25.311-b11, mixed mode)
```



```bash
$ wget https://oceandata.sci.gsfc.nasa.gov/SeaDAS/installer/7.3.1/seadas_7.3_linux64_installer.sh

$ bash seadas...sh

```

这个是最后一个不需要装GUI的版本，怪不得看那么多论文一直都在用7.3.

Where should SeaDAS be installed?
[/opt/seadas-7.3]

>一个奇怪的地方
>
>我第一次发现linux的路径居然是这样的
>
>就是cd /进入的才是根目录
>
>这个目录是所有用户共享的目录
>
>而cd ~进入的是我自己的目录
>
>学习了

## OCSSW

最重要的部分

```bash
$　wget https://oceandata.sci.gsfc.nasa.gov/ocssw/install_ocssw.py
$ chmod +x install_ocssw.py
$ ./install_ocssw.py --install-dir=/root/SeaDAS/seadas-7.3/ocssw --aqua --seawifs --goci
$ mv /opt/seadas-7.3 ~/SeaDAS
$ vim ~ .bashrc
```

输入

```bash
export SEADASPATH=/root/SeaDAS/seadas-7.3
export PATH=$PATH:$SEADASPATH/bin
```

where: `[SeaDAS_install_dir]` is the directory where you installed SeaDAS.

搞完之后保存就行

到这之后我忽然发现我在OCSSW路径下找不到那几个装好的文件，搜了一下果然是给装到/而不是~，自己上面那个下载地方能改路径但是自己忘了改



只能重新移动过来

测试命令

```bash
$ source .../OCSSW_bash.env
$ which l2gen
```

显示

```bash
/root/SeaDAS/seadas-7.3/ocssw/bin/l2gen
```

就是搞定了

然后加了个更全一点的东西放在bashrc里

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

## 文件传输

之前都是直接用Putty或者filezelia直接弄，问了下公司程序员说要用SCP

```bash
$ scp SourceFile user@server:directory/TargetFile
```

再传回来就是

```bash
$ scp user@server:directory/SourceFile TargetFile
```

## 大气校正

因为NASA数据格式的更新，和之前的那个脚本稍微有些不一样

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



### 车了个又翻

跑了个测试数据之后发现跑不出来

只能重装

然后这个网吧 又不稳定

![image-20211209155834683](C:/Users/zhenjia/AppData/Roaming/Typora/typora-user-images/image-20211209155834683.png)

经典404

查了查发现是他们官网维护

##　车了个再翻回来

```bash
$ bzip2 -d *.bz2
$ ls -1 *L1A_LAC* > input.txt
$ bash l1_to_l2.sh inupt.txt
```



