---
title: matplotlib xarray basemap h5py conflict
tags:
  - python
categories: Study Notes
abbrlink: 5ce7c25
date: 2021-12-06 17:04:26
mathjax:
copyright:
password:
lang: en
---

A problem that troubled me for a week.

<!-- more -->

Recently I've been encountering two issues frequently.

One is:

```bash
ImportError: cannot import name 'dedent
```

The other is h5py version issues.

Environment: python 3.7.12

After searching for a while, I finally understood that xarray (h5py) requires matplotlib >= 3.5, while basemap requires matplotlib <= 3.2 (that's the dedent issue). If you force downgrade for basemap, h5py won't work. If you don't downgrade, basemap won't work.

If you let conda auto-resolve conflicts, xarray and h5py versions become incompatible again.

My final solution was to add the following code to the init.py file in E:\ana\envs\weattech\Lib\site-packages\matplotlib\cbook:

```python

_dedent_regex={}

@deprecated("3.1", alternative="inspect.cleandoc")
def dedent(s):
    """
    Remove excess indentation from docstring *s*.
    """
    if not s:
        return ''

    match = _find_dedent_regex.match(s)
    if match is None:
        return s

    nshift = match.end(1) - match.start(1)
    if nshift == 0:
        return s

    unindent = _dedent_regex.get(nshift, None)
    if unindent is None:
        unindent = re.compile("\n\r? {0,%d}" % nshift)
        _dedent_regex[nshift] = unindent

    result = unindent.sub("\n", s).strip()
    return result
```

After this, you'll see this warning when using it:

```bash
E:\ana\envs\weattech\lib\site-packages\pyresample\bilinear\__init__.py:50: UserWarning: XArray and/or zarr not found, XArrayBilinearResampler won't be available.
```

But it mostly works now.
