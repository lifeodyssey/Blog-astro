---
title: matplotlib xarray basemap h5py コンフリクト
tags:
  - python
categories: 学習ノート
abbrlink: 5ce7c25
date: 2021-12-06 17:04:26
mathjax:
copyright:
password:
lang: ja
---

1週間悩まされた問題。

<!-- more -->

最近よく2つの問題に遭遇します。

1つ目：

```bash
ImportError: cannot import name 'dedent
```

もう1つはh5pyのバージョン問題。

環境：python 3.7.12

しばらく調べて、やっと分かりました。xarray（h5py）はmatplotlib >= 3.5を要求し、basemapはmatplotlib <= 3.2を要求します（dedentの問題）。basemapのためにダウングレードするとh5pyが使えなくなり、ダウングレードしないとbasemapが使えません。

condaに自動でコンフリクトを解決させると、xarrayとh5pyのバージョンが再び互換性がなくなります。

最終的な解決策は、E:\ana\envs\weattech\Lib\site-packages\matplotlib\cbookのinit.pyに以下のコードを追加することでした：

```python
_dedent_regex={}

@deprecated("3.1", alternative="inspect.cleandoc")
def dedent(s):
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

これで使用時にこの警告が出ます：

```bash
UserWarning: XArray and/or zarr not found, XArrayBilinearResampler won't be available.
```

でも大体は使えるようになりました。
