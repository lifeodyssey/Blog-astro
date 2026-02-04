---
title: Jupyter Lab
tags:
  - python
categories: Study Notes
mathjax: true
abbrlink: c958af43
date: 2021-05-12 11:25:46
copyright:
lang: en
---

I can only write code in Jupyter for now, but my original notebook kept getting bigger and slower because I kept writing multi-megabyte files, so I switched to Lab.

Here are some configurations I set up while using it.

<!-- more -->

# Extension

(Starting with confusion - I'm actually following a [Japanese blog](https://qiita.com/canonrock16/items/d166c93087a4aafd2db4) to set this up, couldn't find Chinese resources)

You need to install NodeJS to install extensions (why?)

```bash
conda install -c conda-forge nodejs
```

Then you can use List to see what's installed:

```
jupyter labextension list
```

I mainly installed a few from [Awesome Jupyter](https://github.com/markusschanta/awesome-jupyter), basically to make it similar to the original Notebook, like lsp, toc, drawio, latex, variableinspector. Others not in that list include [jupyterlab-execute-time](https://github.com/deshaw/jupyterlab-execute-time) and [jupyterlab-kite](https://github.com/kiteco/jupyterlab-kite).

Finally run this:

```bash
jupyter lab build
jupyter labextension enable all
```

Or manually enable extensions in Lab's settings.

One inconvenience is that the Variable inspector can't float.

One more thing - as a casual user, I feel kite isn't necessary, lsp is enough.

That said, I still feel notebook is more convenient.
