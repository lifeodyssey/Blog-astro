---
title: WSL+anaconda+jupyter
tags:
  - 机器学习
  - Linux
categories: 学习笔记
copyright: true
abbrlink: 60b81fb7
date: 2021-12-04 19:00:45
mathjax:
password:
---



自己之前经常用autosklearn，但是这玩意在windows下不能跑，本来想跟之前一样装个ubuntu但是感觉太麻烦了，于是试着用WSL，这里是配置全过程

<!-- more -->

# WSL

以管理员权限打开cmd，输入

```bash
$ wsl --install
```

重启之后在MS store里下载ubuntu就行 我下了unbuntu 20.04LTS

然后在命令行里直接输入wsl就可以开始了

![image-20211204211035801](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewinimage-20211204211035801.png)

注意这个是默认安装在C盘。

补一个[ubuntu常用命令](https://www.cnblogs.com/linuxws/p/9307187.html)

# anaconda

我这里装的是miniconda 感觉自己也用不了那么多包

命令如下

```bash
$ mkdir -p miniconda
$ cd miniconda
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

然后我就卡在这一步了 搞了半天都弄不过去

只能去清华镜像下了miniconda，注意下py37版本的，然后直接复制到miniconda这里面。文件目录一般是c/users/username/miniconda

然后

```bash
$ bash Miniconda.... .sh
$ conda create --name pysoc python=3.7
$ conda activate pysoc
```

然后就是必不可少的社会主义特色，换清华。

## 插曲

在那之前还有个小插曲。

我想直接用reboot来重启，但是不行，显示

```bash
$ System has not been booted with systemd as init 
$ system (PID 1). Can't operate.
$ Failed to connect to bus: Host is down
$ Failed to talk to init daemon.
```

查了一堆，虽然我没懂到底是啥问题，但是我最后是这么解决的

```bash
$ sudo -s
$ wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
$ sudo dpkg -i packages-microsoft-prod.deb
$ sudo apt-get update
$ apt search dotnet-sdk
```

这里是为了解决直接安装dotnet-sdk出现的Unable to locate package dotnet-sdk这个错误

然后接着

```bash
$ sudo apt install -y daemonize dbus dotnet-runtime-5.0 gawk libc6 libstdc++6 policykit-1 systemd systemd-container

```

装一些必要的module

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

设置wsl-transdebian的repo

```bash
$ sudo apt install -y systemd-genie
$ genie -i
$ genie -s
$ genie -l
```

装并且使用genie

然后成功翻车了

查了查原因是

>So why don't normal `reboot`/`shutdown` commands work? Two reasons. First, as covered by Bengt, WSL doesn't currently support systemd, and Ubuntu simply links these two legacy commands to `/usr/bin/systemctl` (the systemd control utility). You can see this with `ls -l /usr/sbin`.
>
>But even if these were the legacy commands which directly called [Linux's shutdown API](https://stackoverflow.com/q/28812514/11810933), it wouldn't work. Microsoft doesn't typically hook up API's that interact directly with the hardware, instead providing virtualization interfaces where necessary. But in the case of starting and stopping a WSL instance, it's just so "lightweight" (as discussed above) that there's not any real reason to do so.

最后只能另开一个cmd

```bash
$ wsl --list
$ wsl --terminate Ubuntu
#or wsl shutdown to shut all the sub system
```

重启进入之后显示前面有个base就是进了conda环境了

## 换源

```bash
$　conda config --add channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/'
$ conda config --set show_channel_urls yes
```

## 搞环境

```bash
$ conda create --name pysoc python=3.7 
$ conda activate pysoc
```

千万 千万 千万不要在base里跑东西装东西

经典装包时间

```bash
$ conda install gxx_linux-64 gcc_linux-64 swig
$ conda config --add channels conda-forge
$ conda config --set channel_priority strict
$ conda install auto-sklearn
$ conda install -c conda-forge geopandas
```

然后成功再次翻车

![image-20211205111723183](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewinimage-20211205111723183.png)

试着删除再重新创建环境也不行，没办法，重装系统吧

直接打开ubuntu的设置点击reset就行

![image-20211205114045880](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewinimage-20211205114045880.png)

重装之后再来一遍，不过这次没有搞reboot相关的那个包genie，

然后这次就没有翻车了，装完上面那俩之后，继续

```bash
$ conda install seaborn
$ conda install -c conda-forge xarray dask netCDF4 bottleneck
$ conda install -c conda-forge cartopy
$ conda install -c conda-forge pyresample
$ conda install python-ternary
```

常用的好像就这些，有啥缺少的待会再试试。

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

可以参考之前写的[这个](https://lifeodyssey.github.io/posts/c958af43.html)

然后直接输入jupyter lab就行了

由于可能闪过一大堆信息，直接在cmd里跳到下一页，导致看不到token，可以摁一下ctrl+c，会出来token，复制到浏览器就行

因为我git clone的repo也在c盘，就直接打开直接之前写的jupyterlab用就行了，

总共花了我七八个小时吧，大部分都是因为想搞reboot弄出来的 没那玩意大概四个小时结束。

注意访问的路径改变为/mnt/c(defgh...)

## 再次一个车的翻

我在jupyter里尝试import autosklearn的时候显示autosklearn没有，原来conda 那句没有装上

再次试着装了下，我发现，居然是geopandas和autosklearn有冲突，我无语了。

最后用

```bash
$ pip3 install auto-sklearn
```

搞定了

另外注意autosklearn的一些关键字现在（2021.12.5）和我本科的那个有一些区别，注意就行。

# CPU and memory limit

跑了一天之后发现还没跑出来结果，刷新一看显示kernel seems to die。

然后查了查是分配的内存不够

```bash
$ wsl --shutdown
$ notepad "$env:USERPROFILE/.wslconfig"
```

写入

[wsl2]
memory=3GB   # Limits VM memory in WSL 2 up to 3GB
processors=4 # Makes the WSL 2 VM use two virtual processors

想写多少写多少

在wsl里用free -h可以看到内存分配情况

# 一个bug



最近碰到了无法识别外接硬盘的问题

就是老显示No such device

然后查到了[这个](https://dowww.spencerwoo.com/)

真的是宝藏！快去给我收藏。

虽然这个问题跟我遇到的并不一样

## 密码重置

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

