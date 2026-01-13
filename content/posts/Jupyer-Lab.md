---
title: Jupyter Lab
tags:
  - python
categories: 学习笔记
mathjax: true
abbrlink: c958af43
date: 2021-05-12 11:25:46
copyright:
---

本菜鸡到现在也只能拿Jupyter写代码，但是原来的notebook因为我动不动就写个好几M的玩意越来越大越来越卡，所以我就换成Lab了。

这里讲一些用他的时候搞得一些配置啥的

<!-- more -->

# Extension

(开头黑人问号，我居然是在参照一个[日语的博客](https://qiita.com/canonrock16/items/d166c93087a4aafd2db4)来搞这个，居然找不到中文的)

安装Extension必须得安装NodeJS(黑人问号❓)

```bash
conda install -c conda-forge nodejs
```

然后可以拿List看装了啥

```
jupyter labextension list
```

我主要装了[Awesome Jupyter](https://github.com/markusschanta/awesome-jupyter)里面的几个，主要就是把它搞得跟原来的Notebook差不多就行，比如lsp,toc,drawio,latex,variableinspector。不在这里面的有[jupyterlab-execute-time](https://github.com/deshaw/jupyterlab-execute-time)，[jupyterlab-kite](https://github.com/kiteco/jupyterlab-kite)

最后来一句这个

```bash
jupyter lab build
jupyter labextension enable all
```



或者在lab里的setting里手动开启extention

感觉不太方便的地方就是那个Variable inspector不能float

多说一句，纯路人，感觉kite没啥必要,lsp足够了

虽然但是，还是感觉notebook好用啊