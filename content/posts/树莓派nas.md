---
title: 树莓派nas-nextcloudpi
tags:
  - 树莓派
  - NAS
  - 摄影处理
categories: 自己动手
abbrlink: d7f3beca
date: 2022-07-02 15:15:14
mathjax:
copyright:
---

今年年初就买了树莓派，本来想拿他做一个自动摄影分析的东西，分析一下天气来跑深度学习，但是一直很忙没开机。目前看来对我来说做NAS的需求更强烈，所以就拿来做NAS了。

我对于树莓派应该会有以下的几个需求

- NAS 用来存放相机和手机的照片，方便P图整理备份
- 自动下载电影和番剧
- 电子图书馆
- homekit

这里先写第一个，希望能赶紧解决吧哈哈

<!-- more -->

# 系统烧录与设置

看了一堆中文的帖子都是在raspi上设置nextcloud服务，过程复杂，其实最简单的方法是直接用nextcloud pi，下载地址在https://github.com/nextcloud/nextcloudpi/releases/tag/v1.47.2

不过注意这个只是一个还没测试过的版本，具体有啥坑俺也不知道

基本根据这个来的https://www.makeuseof.com/raspberry-pi-nextcloud/ 除此之外参考的文档在这https://nextcloudpi-documentation.readthedocs.io/en/latest/

安装完之后只有命令行。

用lsb_release -a看了下 是Debian GNU/Linux 11(bullseye)

ip address show看了下ip

sudo service ssh start 启动ssh服务

输入 sudo ncp-config 并在提示更新时选择是。 在下一个菜单中，选择 CONFIG 并使用向下箭头滚动到列表底部。 在这里，选择 nc-webui，然后删除 no 并输入 yes。 按 Enter，然后按任意键。 选择 Back and Finish 退出配置工具。

除此之外为了方便我用ssh 我在network里还开启了config，这里需要 ssh-keyscan -t rsa server_ip　

在另一台计算机上的网络浏览器中，输入地址 https://[您的 Pi 的 IP 地址]:4443

如果您看到您的连接不私密或不安全的警告，请选择忽略它（通过在 Chrome 或 Firefox 中选择高级）并继续访问该站点。

系统会提示您登录。默认用户名是 ncp，密码是 ownyourbits。

NextCloudPi 激活屏幕显示两个密码，您需要将其复制并粘贴到文档中以安全保存。 第一个是 NextCloudPi Web 面板的密码，可让您配置服务器设置。 第二个是 Nextcloud Web 界面本身。 如果需要，您可以稍后更改这些密码。

记下这些密码后，选择激活。 几秒钟后，系统将提示您登录 NextCloudPi 网络面板。 在这里可以对nextcloudpi进行一些设置 点击右上角一个魔法棒可以很方便的进行修改。

转到 https://[您的 Pi 的 IP 地址]（不带 :4443 后缀）并使用用户名 ncp 和您记下的第二个密码登录。

通过欢迎屏幕后，您将看到主 Web 仪表板。 这是您自己的树莓派 4 云服务器！

# 修改pi设置

这一步甚至简单到不需要命令行。

fdisk -l看了眼硬盘

![image-20220706113025038](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202207061130124.png)

https://support.huaweicloud.com/evs_faq/evs_faq_0022.html

nc-automount

https://zhuanlan.zhihu.com/p/258913077

https://oscarcx.com/tech/raspberrypi-nas.html

https://cloud.tencent.com/developer/article/1857115

https://blog.csdn.net/qq_35566365/article/details/122536163

https://cloud-atlas.readthedocs.io/zh_CN/latest/arm/raspberry_pi/startup/pi_os.html

https://www.openmediavault.org/
