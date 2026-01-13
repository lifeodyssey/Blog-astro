---
title: 服务器部署自动机器学习
copyright: true
tags:
  - 机器学习
  - Linux
  - 服务器
categories: 学习笔记
abbrlink: c23e0e6e
date: 2019-06-05 14:25:11
mathjax:
---

借着毕业实习的机会将代码放到了实验室的服务器上，在这里记录一下全过程。

<!-- more -->

# 实验室服务器的登陆及基本操作

除了直接在服务器主机上登陆之外，还可以通过一些连接软件进行远程登陆。我这里使用的是PuTTY软件。

PuTTY是一款开源(Open Source Software)的连接软件，主要由Simon Tatham维护，使用MIT许可证授权。包含的组件有：PuTTY, PuTTYgen,PSFTP, PuTTYtel, Plink, PSCP, Pageant,默认登录协议是SSH，默认的端口为22。Putty是用来远程连接服务器的，支持SSH、Telnet、Serial等协议的连接。其中最常用的是SSH。用它来远程管理Linux十分好用，其主要优点如下：

- 完全免费开源;
- 全面支持windows系统;
- 全面支持SSH1和SSH2；
- 绿色软件，无需安装，下载后在桌面建个快捷方式即可使用；
- 体积很小，不到1M；
- 操作简单，所有的操作都在一个控制面板中实现。

PuTTY的下载页面为<https://putty.org/>，选择合适的版本下载安装即可。

下载好之后在Saved Sessions处输入服务器IP地址，点击Save，然后点Open即可打开会话，输入用户名和密码即可登陆。

 ![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20190605143842.png)

接着使用一些基本命令来确定服务器的系统版本和python版本。

输入

```bash
cat /etc/redhat-release
```

 返回的系统版本为Centos 7.3.1611 Redhat。

输入

```bash
python
```

返回的系统版本为python2.7。

利用mkdir命令新建一个目录，后续操作皆在此目录下进行。

```bash
mkdir amf
```

CPUE自动预测系统的环境配置

实验室所用linux服务器版本为Centos 7.3.1611 Redhat，默认python版本为2.7，因此首先需要将python3安装到服务器上。

# python3的安装

## RPM包安装

本次安装使用的是来源于IUS社区的RPM包进行安装。IUS是“Inline with Upstream Stable”的缩写，他主要是一个提供新版本RPM包的社区，具体情况可以查看[官方文档](https://ius.io/GettingStarted/#install-via-automation)。

所使用的具体操作如下(部分操作需要sudo权限，在这里不一一列出)：

(1)   添加IUS地址：

```bash
yum -y install https://centos7.iuscommunity.org/ius-release.rpm
```

(2)   创建缓存元数据：

```bash
yum makecache


```

(3)   安装python3.6：

```bash
yum install python36u

yum -y install python36u-pip

yum -y install python36u-devel
```

（4）测试环境：输入python3.6出现如下文字即代表安装成功。

```python
Python 3.6.8 (default, May  2 2019, 20:40:44)

[GCC 4.8.5 20150623 (Red Hat 4.8.5-36)] on linux

Type "help", "copyright", "credits" or "license" for more information.
```

## 虚拟环境的配置

因为系统中存在多个python版本，为了避免环境污染，我使用virtualenv创建了独立的虚拟环境，具体过程如下：

```bash
python3.6 -m venv py3

source py3/bin/activate
```

执行完之后再命令行前会出现(py3)的字样，即代表进入虚拟环境中，输入python -V返回python3.6.8即代表安装成功。

## 机器学习环境配置

自动预测系统是基于auto-sklearn开发的，需要scipy、sci-kit、pands、numpy等代码库的支持。在上一不中我们安装了pip这一python包管理工具，它可以提供对python包的查找、下载、安装、卸载等功能，因此这里直接采用pip命令进行安装即可

需要注意的是，在配置之前要先进入上一步配置的虚拟环境。

依次执行如下命令即可进行安装。

```bash
pip install pandas

pip install scipy

pip install scikit-learn

pip install auto-sklearn

pip install matplotlib

pip install xlrd

pip install openpyxl
```

在安装auto-sklearn的过程中遇到如下错误：

```bash
Error: Syntax error in input(3).     error: command 'swig' failed with exit status 1
```

 

根据提示可知是由于swig而出现的错误，SWIG本质上是个代码生成器，为C/C++程序生成到其他语言的包装代码(wrapper code)，这些包装代码里会利用各语言提供的C API，将C/C++程序中的内容暴露给相应语言。为了生成这些包装代码，SWIG需要一个接口描述文件，描述将什么样的接口暴露给其他语言。  

查看swig所用的版本为2.0.10，为python2.7所对应的版本，因此需要在系统中将swig升级到3版本。

首先将swig2卸载，执行命令：

```bash
yum -y remove swig
```

然后安装swig 3，执行命令：

```bash
yum install swig3
```

安装以后执行命令

```bash
swig -version
```

可以看到swig的版本变成了3.0.12。

重新执行pip install auto-sklearn即可完成安装。

安装完成后在虚拟环境中输入python，进入python语言，执行如下命令，如果没有报错，即说明环境配置完成。按Ctrl+D即可退出虚拟环境和puTTY。

```python
import autosklearn.regression

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np
```

# 服务器与windows之间的文件传输

在运行程序之前，需要将要用到的数据传输到服务器上，这里采用的是puTTY提供的PSFTP工具。在之前安装puTTY时已经一并安装了。

打开PSFTP，输入open 服务器地址，完成登陆，然后采用put命令将本地文件传输到服务器上，get命令将服务器上的文件取回。

# 程序调试

利用PSFTP将源代码传输到服务器上，进入虚拟环境，在命令行中输入

```bash
python code.py
```

即可看到程序开始运行。

如果程序报错，需要利用系统自带的vim编辑器进行编辑与调试。vim 共分为三种模式，分别是命令模式（Command mode），输入模式（Insert mode）和底线命令模式（Last line mode）。 这三种模式的作用分别是：

（1）命令模式：

用户刚刚启动 vim，便进入了命令模式。此状态下敲击键盘动作会被Vim识别为命令，而非输入字符。比如我们此时按下i，并不会输入一个字符，i被当作了一个命令。

以下是常用的几个命令：

- l  i 切换到输入模式，以输入字符。
- l  x 删除当前光标所在处的字符。
- l  : 切换到底线命令模式，以在最底一行输入命令。

若想要编辑文本：启动Vim，进入了命令模式，按下i，切换到输入模式。

命令模式只有一些最基本的命令，因此仍要依靠底线命令模式输入更多命令。

（2）输入模式：

在命令模式下按下i就进入了输入模式。在输入模式中，可以使用以下按键：

- l  字符按键以及Shift组合，输入字符
- l  ENTER，回车键，换行
- l  BACK SPACE，退格键，删除光标前一个字符
- l  DEL，删除键，删除光标后一个字符
- l  方向键，在文本中移动光标
- l  HOME/END，移动光标到行首/行尾
- l  Page Up/Page Down，上/下翻页
- l  Insert，切换光标为输入/替换模式，光标将变成竖线/下划线
- l  ESC，退出输入模式，切换到命令模式

（3）底线命令模式：

在命令模式下按下:（英文冒号）就进入了底线命令模式。底线命令模式可以输入单个或多个字符的命令，可用的命令非常多。在底线命令模式中，基本的命令有（已经省略了冒号）：

- l  q 退出程序
- l  w 保存文件
- l  按ESC键可随时退出底线命令模式。

如果你想用vim来编辑或者建立一个名为test.txt的文件时，只需要在虚拟环境中输入：

```bash
vim test.txt
```

即可。此时进入的是一般模式，在一般模式下按i即可进入输入模式，此时左下角状态栏会出现–INSERT- 的字样，即代表可以输入任意字符。这时除了ESC之外，其余按键都可以作为一般的输入按钮。编辑完成之后按ESC即可挥动一般模式。在一般模式下按：，输入wq即可保存离开。

除此之外还有很多进阶命令，在这里就不再细数了。

 