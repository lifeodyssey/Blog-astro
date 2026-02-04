---
title: Publishable plot using python&R
tags:
  - Research
  - python
  - R
categories: 学習ノート
mathjax: true
abbrlink: cd046c9b
date: 2020-12-01 19:45:26
copyright:
lang: ja
---

これは長い記事になります。

どれくらい時間がかかるか分からないほど長くなります。

<!-- more -->

matplotlibに苦しんでいる人は多いです。

今月（2020年12月）中に完成させるという目標を立てます。

主に以下の種類のプロットを含みます：

1. 観測点マップ

2. 海洋環境要素分布図

3. 散布図

4. 密度散布図

5. 棒グラフ

6. CI付き折れ線グラフ

   最初のものはRを使用し、残りはデータ処理時に直接呼び出す必要があるためPythonを使用します。

   ただし、2番目については結果をncファイルに出力してからRで描画することも検討しています。

   また、これは主に自分用のコードメモなので、中国語は使いません。

# 散布図

```python
def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength > 750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return np.array([R, G, B, A]).reshape(1, -1)
```

```python
def single_color_scatter(ax1, x, y, xlable: str, ylable: str, xlim: list = None, title: str = None, ylim: list = None , color: str = 'k', ):
  ##import packages
  import seaborn as sns
  import matplotlib.patches as mpl_patches
  from sklearn.metrics import mean_squared_error
  sns.set_style('white')

  ## do regression and get metrics
  max_lim = np.max([np.max(x), np.max(y)])
  min_lim = np.min([np.min(x), np.min(y)])
  x_11 = np.linspace(min_lim, max_lim)
  y_11 = x_11
  N = len(x)

  result = regress2(x.flatten(), y.flatten())
  Slope = result['slope']
  Intercep = result['intercept']
  y_fit = Slope * x + Intercep
  rmse = round(np.sqrt(mean_squared_error(x.flatten(), y.flatten())), 5)
  r = round(result['r'], 5)
  bias = (np.sum((y - x) / x)) / N

  # start plot
  line11, = ax1.plot(x_11, y_11, color='k', linewidth=1.5, linestyle='--', label='1:1 line', zorder=5)
  linefit, = ax1.plot(x, y_fit, color='r', linewidth=2, linestyle='-', label='fitted line', zorder=5)

  ax1.scatter(x, y, edgecolor=None, c=color, s=50, marker='s', facecolors="None", zorder=3)
  fontdict1 = {"size": 30,
               "color": 'k',
               'family': 'Time New Roman'}

  ax1.set_xlabel(xlable, fontdict=fontdict1)
  ax1.set_ylabel(ylable, fontdict=fontdict1)
  ax1.grid(False)
  l0 = ax1.legend(handles=[line11, linefit], loc='lower right', prop={"size": 25})
  # set tick font
  labels = ax1.get_xticklabels() + ax1.get_yticklabels()
  [label.set_fontname('Time New Roman') for label in labels]
  for spine in ['top', 'bottom', 'left', 'right']:
      ax1.spines[spine].set_color('k')
  ax1.tick_params(left=True, bottom=True, direction='in', labelsize=30)
  # add title
  titlefontdict = {"size": 40,
                   "color": 'k',
                   'family': 'Time New Roman'}
  ax1.set_title(title, titlefontdict, pad=20)
  h1fontdict = {"size": 25, 'weight': 'bold'}

  handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                   lw=0, alpha=0)] * 6
  text = [r'$r:$' + str(r),
          r'$RMSE:$' + str(rmse),
          r'$Slope:$' + str(round(Slope, 3)),
          r'$Intercept:$' + str(round(Intercep, 3)),
          r'$Bias:$' + str(round(100 * bias, 3)) + '%',
          r'$N:$' + str(N)]
  l1 = ax1.legend(handles, text, loc='upper left', fancybox=True, framealpha=0, prop=h1fontdict)

  h2text_font = {'size': '10', 'weight': 'medium'}
  label_font = {'size': '10', 'weight': 'medium'}

  orderhand = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                     lw=0, alpha=0)]
  ax1.add_artist(l1)
  ax1.add_artist(l0)
  if xlim is not None:
      ax1.set_xlim(xlim)
  else:
      ax1.set_xlim(min_lim, max_lim)
  if ylim is not None:
      ax1.set_ylim(ylim)
  else:
      ax1.set_ylim(min_lim, max_lim)
```




