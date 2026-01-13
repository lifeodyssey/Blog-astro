---
title: lfs
tags:
  - python
  - git
categories: 学习笔记
mathjax: true
abbrlink: 442707f1
date: 2021-11-29 12:06:20
copyright:
password:
---

Github大文件

<!-- more -->

最近在搞的时候经常碰见

fatal: sha1 file '<stdout>' write error: Broken pipe

这样的错误 

其实原因就是因为github只支持最大一个100mb，平时写代码够用了，但是如果有那么几张hdf或者nc啥的就不行了。

装好lfshttps://git-lfs.github.com/之后用命令行(pycharm好像用不了)就可以了。

其实最好的办法就是把数据文件分开存储，但是我中间一次commit把文件也跟加上了，再换路径也会被push过去。

就只好先传上去转移路径 然后commit push

而且lfs也是有限制的 只能传1g，剩下的就要付费。就还真的挺麻烦的。

目前没找到更好的办法。

## [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)

更好的办法说来就来了

是在[这个页面](https://stackoverflow.com/questions/2100907/how-to-remove-delete-a-large-file-from-commit-history-in-the-git-repository)找到的。

如果你已经把一部分文件上传到gayhub了，可以下载bfg。

然后

```bash
$ git clone --mirror git://example.com/some-big-repo.git
$ java -jar bfg.jar --strip-blobs-bigger-than 100M my-repo.git
$ git gc --prune=now --aggressive
```

搞定

需要注意的是这个方法需要有jre

##　git-filter-branch

但是上面那个办法对我来讲是一个死循环

因为文件太大，没法push到gayhub

所以就没办法clone那个mirror再clean commit

另一个办法是

```bash
$ git filter-branch -f --index-filter "git rm -rf --cached --ignore-unmatch *.HDF" -- --all
```

像我 我知道大文件都是hdf，直接用通配符就行

然后

```bash
$ rm -rf .git/refs/original/
$ git reflog expire --expire=now --all
$ git gc --prune=now
$ git gc --aggressive --prune=now
```

最后把他们全部push过去

```bash
$ git push --all --force
```

