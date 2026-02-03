---
title: Obsidian和hexoblog的多平台同步（synchronization for blog and obisidian）
draft: true
tags:
  - 随便写写
  - 博客搭建
categories: 资料贴整理
abbrlink: 1d4ba56a
date: 2021-10-07 15:13:19
mathjax:
copyright:
password:
---

一个解决方案记录贴

<!-- more -->

因为自己现在搞了俩电脑，虽然原本的打算是游戏本只用来跑程序和打游戏，不安装什么乱七八糟的微信之类的，所有的文案类工作都放在另一个轻薄本上，但是现在看来还是有点难，比如我想一遍写程序一遍把程序放到博客上。现在打算是游戏本上只处理轻文本，以markdown为主，通过obisidian同步过去，在另一个电脑上写出word和ppt做最终处理。

其实这俩跨平台同步最好的就是利用第三方的服务，比如之前我就把[zotero](https://lifeodyssey.github.io/posts/4ce7830b.html)放在了onedrive里，只需要改一下储存的位置就好了。但是一方面我俩电脑上的node和hexo的版本好像不太一样，插件会有冲突，另一方面就是熟练一下我git的操作，所以就写这个来记录一下。

# Obsidian

这个的同步我之前就用的github，这里	其实就直接用github的方式来同步就好，很方便。

## 设置git ssh

下载好git之后打开bash

依次

```bash
git config --global user.name yourname
git config --global user.email youremail
ssh-keygen -t rsa -C your email
```

然后在github里提交ssh就好

## 同步到本地

```bash
git clone
cd
git init
git remote add origin link
git pull origin main
```

## 重装插件

我也记不清原来mac上装了啥了，有点印象的直接装上了

依次有

Better world count

Calendar

Citations

Dataview

Day Planner

Mind Map

Obsidian GIt

Outliner

Periodic Notes

Templater

结束

## some problem you know

I don’t know whether use English or Chinese or Japanese is better to protect my privacy for providing this method. But it actually not very difficult. So please read [this](https://gitee.com/ineo6/hosts)

# hexo

这个稍微麻烦一些 因为平时推送到github pages上面的没有源码，只有渲染好之后的html。目前比较通用的方式是[这种](https://blog.csdn.net/godread_cn/article/details/118383963?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.no_search_link&spm=1001.2101.3001.4242)。就是在github上再建一个分支然后把源文件都搞上去。但是这样对我来讲有个问题就是，我有一部分文章是加密的，传源文件代表着这一部分不再被保护了，最后我还是用了坚果云来搞。

这样唯一的坏处就是无法用hexo d -g在游戏本上发布了，还得同步到另一台电脑上才行，另外就是无法利用hexo new来立马生成一个新的文档。

所以大概就是用他来写草稿，然后放在draft下面，然后等同步回来拿hexo publish

