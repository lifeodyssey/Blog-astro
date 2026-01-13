---
title: matplotlib xarray basemap h5py conflict
tags:
  - python
categories: 学习笔记
abbrlink: 5ce7c25
date: 2021-12-06 17:04:26
mathjax:
copyright:
password:
---

一个困扰了我一星期的问题。



<!-- more -->

最近经常出现两个问题

一个是

```bash
ImportError: cannot import name 'dedent
```

另一个则是h5py的版本问题。

环境为python 3.7.12

找了半天终于明白，原来xarray(h5py)对matplotlib的要求是大于3.5，而basemap的要求则是小于等于3.2(就是那个dedent)。如果强行给basemap降版本会导致h5py用不了，如果不降吧basemap又用不了。

如果用conda自动解决冲突的话，会导致xarray和h5py的版本又不匹配了

我最终的解决方法是，在E:\ana\envs\weattech\Lib\site-packages\matplotlib\cbook里面的init.py里加上下面这段

```python

_dedent_regex={}

@deprecated("3.1", alternative="inspect.cleandoc")
def dedent(s):
    """
    Remove excess indentation from docstring *s*.

    Discards any leading blank lines, then removes up to n whitespace
    characters from each line, where n is the number of leading
    whitespace characters in the first line. It differs from
    textwrap.dedent in its deletion of leading blank lines and its use
    of the first non-blank line to determine the indentation.

    It is also faster in most cases.
    """
    # This implementation has a somewhat obtuse use of regular
    # expressions.  However, this function accounted for almost 30% of
    # matplotlib startup time, so it is worthy of optimization at all
    # costs.

    if not s:  # includes case of s is None
        return ''

    match = _find_dedent_regex.match(s)
    if match is None:
        return s

    # This is the number of spaces to remove from the left-hand side.
    nshift = match.end(1) - match.start(1)
    if nshift == 0:
        return s

    # Get a regex that will remove *up to* nshift spaces from the
    # beginning of each line.  If it isn't in the cache, generate it.
    unindent = _dedent_regex.get(nshift, None)
    if unindent is None:
        unindent = re.compile("\n\r? {0,%d}" % nshift)
        _dedent_regex[nshift] = unindent

    result = unindent.sub("\n", s).strip()
    return result
```



这样再次使用的时候会出现一个这个

```bash
E:\ana\envs\weattech\lib\site-packages\pyresample\bilinear\__init__.py:50: UserWarning: XArray and/or zarr not found, XArrayBilinearResampler won't be available.
  warnings.warn("XArray and/or zarr not found, XArrayBilinearResampler won't be available."
```

但是大体上不影响使用了
