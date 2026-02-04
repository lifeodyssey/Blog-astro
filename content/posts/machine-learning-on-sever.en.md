---
title: Deploying AutoML on Server
copyright: true
tags:
  - Machine Learning
  - Linux
  - Server
categories: Learning Notes
abbrlink: c23e0e6e
slug: deploying-automl-on-server
date: 2019-06-05 14:25:11
lang: en
mathjax:
---

Taking advantage of my graduation internship, I deployed my code to the lab server. Here I'll document the entire process.

<!-- more -->

# Lab Server Login and Basic Operations

Besides logging in directly on the server host, you can also use connection software for remote login. I'm using PuTTY here.

PuTTY is an open source connection software, mainly maintained by Simon Tatham, licensed under the MIT license. It includes components: PuTTY, PuTTYgen, PSFTP, PuTTYtel, Plink, PSCP, Pageant. The default login protocol is SSH with default port 22. PuTTY is used for remote server connections, supporting SSH, Telnet, Serial and other protocols. SSH is the most commonly used. It's very useful for remote Linux management, with the following main advantages:

- Completely free and open source
- Full Windows support
- Full SSH1 and SSH2 support
- Portable software, no installation needed, just create a desktop shortcut after download
- Very small size, less than 1MB
- Simple operation, all operations in one control panel

PuTTY download page: <https://putty.org/>, select the appropriate version to download and install.

After downloading, enter the server IP address in Saved Sessions, click Save, then click Open to start the session. Enter username and password to log in.

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20190605143842.png)

Then use some basic commands to check the server's system version and Python version.

Enter

```bash
cat /etc/redhat-release
```

The returned system version is CentOS 7.3.1611 Redhat.

Enter

```bash
python
```

The returned Python version is 2.7.

Use the mkdir command to create a new directory. All subsequent operations will be performed in this directory.

```bash
mkdir amf
```

CPUE Automatic Prediction System Environment Configuration

The lab Linux server version is CentOS 7.3.1611 Redhat, with default Python version 2.7, so we first need to install Python 3 on the server.

# Python 3 Installation

## RPM Package Installation

This installation uses RPM packages from the IUS community. IUS stands for "Inline with Upstream Stable", a community that provides newer version RPM packages. For details, see the [official documentation](https://ius.io/GettingStarted/#install-via-automation).

The specific operations used are as follows (some operations require sudo privileges, not listed individually):

(1) Add IUS repository:

```bash
yum -y install https://centos7.iuscommunity.org/ius-release.rpm
```

(2) Create cache metadata:

```bash
yum makecache
```

(3) Install Python 3.6:

```bash
yum install python36u

yum -y install python36u-pip

yum -y install python36u-devel
```

(4) Test environment: Enter python3.6 and if the following text appears, installation is successful.

```python
Python 3.6.8 (default, May  2 2019, 20:40:44)

[GCC 4.8.5 20150623 (Red Hat 4.8.5-36)] on linux

Type "help", "copyright", "credits" or "license" for more information.
```

## Virtual Environment Configuration

Since there are multiple Python versions in the system, to avoid environment pollution, I used virtualenv to create an independent virtual environment. The process is as follows:

```bash
python3.6 -m venv py3

source py3/bin/activate
```

After execution, (py3) will appear before the command line, indicating you've entered the virtual environment. Enter python -V and if it returns python3.6.8, installation is successful.

## Machine Learning Environment Configuration

The automatic prediction system is based on auto-sklearn, requiring support from scipy, scikit-learn, pandas, numpy and other libraries. In the previous step, we installed pip, a Python package management tool that provides functions for finding, downloading, installing, and uninstalling Python packages.

Note: Before configuration, you must first enter the virtual environment configured in the previous step.

Execute the following commands in sequence to install:

```bash
pip install pandas

pip install scipy

pip install scikit-learn

pip install auto-sklearn

pip install matplotlib

pip install xlrd

pip install openpyxl
```

During the installation of auto-sklearn, the following error occurred:

```bash
Error: Syntax error in input(3).     error: command 'swig' failed with exit status 1
```

According to the prompt, this error is caused by swig. SWIG is essentially a code generator that generates wrapper code for C/C++ programs to other languages. These wrapper codes use the C API provided by each language to expose the content of C/C++ programs to the corresponding language.

The swig version was 2.0.10, corresponding to Python 2.7, so we need to upgrade swig to version 3.

First uninstall swig2:

```bash
yum -y remove swig
```

Then install swig 3:

```bash
yum install swig3
```

After installation, execute:

```bash
swig -version
```

You can see the swig version has changed to 3.0.12.

Re-execute pip install auto-sklearn to complete the installation.

After installation, enter python in the virtual environment, enter the Python interpreter, and execute the following commands. If there are no errors, the environment configuration is complete. Press Ctrl+D to exit the virtual environment and PuTTY.

```python
import autosklearn.regression

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np
```

# File Transfer Between Server and Windows

Before running the program, you need to transfer the data to the server. Here I use the PSFTP tool provided by PuTTY, which was already installed with PuTTY.

Open PSFTP, enter "open server_address" to complete login, then use the put command to transfer local files to the server, and the get command to retrieve files from the server.

# Program Debugging

Use PSFTP to transfer the source code to the server, enter the virtual environment, and enter in the command line:

```bash
python code.py
```

You can see the program start running.

If the program reports an error, you need to use the system's built-in vim editor for editing and debugging. Vim has three modes: Command mode, Insert mode, and Last line mode.

(1) Command mode:

When you start vim, you enter command mode. In this state, keyboard actions are recognized as commands, not character input. For example, pressing i won't input a character; i is treated as a command.

Common commands:
- i: Switch to insert mode to input characters
- x: Delete the character at the current cursor position
- :: Switch to last line mode to enter commands at the bottom

(2) Insert mode:

Press i in command mode to enter insert mode. In insert mode, you can use the following keys:
- Character keys and Shift combinations to input characters
- ENTER for new line
- BACKSPACE to delete the character before cursor
- DEL to delete the character after cursor
- Arrow keys to move cursor in text
- HOME/END to move cursor to line start/end
- Page Up/Page Down to scroll up/down
- Insert to toggle between input/replace mode
- ESC to exit insert mode and return to command mode

(3) Last line mode:

Press : (colon) in command mode to enter last line mode. Basic commands (colon omitted):
- q: Quit program
- w: Save file
- Press ESC to exit last line mode anytime

To edit or create a file named test.txt with vim, enter in the virtual environment:

```bash
vim test.txt
```

This enters normal mode. Press i to enter insert mode, and --INSERT-- will appear in the status bar. After editing, press ESC to return to normal mode. In normal mode, press :, enter wq to save and exit.

There are many more advanced commands not covered here.
