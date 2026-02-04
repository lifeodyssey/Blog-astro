---
title: cartopy
copyright: true
tags:
  - Research
  - python
categories: 学習ノート
mathjax: true
abbrlink: c6b8edee
date: 2020-08-01 19:57:05
lang: ja
---

Basemapは2020年末にメンテナンスを終了し、cartopyがその代替となりました。ここにいくつかの学習ノートを書きます。

まず結論から：2020年8月1日時点で、バージョン0.18はまだbasemapを完全に置き換えることができません。特に沿岸データの解像度の問題がありますが、すでに優位性を示しています。今後もっと使ってみる必要があります。

<!-- more -->

# コード例

```python
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
file=r'I:\Nagoya University\Project\Seto\MODIS\20100608T04.nc'
nc=nc.Dataset(file,'r')
lon=nc.groups['navigation_data'].variables['longitude'][:]
lat=nc.groups['navigation_data'].variables['latitude'][:]
variables=nc.groups['geophysical_data'].variables
chl=variables['chlor_a']
minlat = 32.5
minlon = 130.5
maxlat = 35
maxlon = 136
f = plt.figure(figsize=(20, 20), dpi=300)
m = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
f = plt.pcolormesh(lon, lat,chl, shading='flat', vmin=np.log10(0.01), vmax=np.log10(50), cmap=plt.cm.viridis)
m.coastlines(resolution='10m', color='black', linewidth=1)
extent=[130.5,136,32,35]
m.set_extent(extent)
m.add_feature(cfeature.RIVERS)
m.add_feature(cfeature.COASTLINE.with_scale('10m'))
m.add_feature(cfeature.LAND.with_scale('10m'), facecolor='0.75')

g1 = m.gridlines(draw_labels = True)
g1.xlabels_top = False
g1.xlabel_style = {'size': 16, 'color': 'gray'}
g1.ylabel_style = {'size': 16, 'color': 'gray'}
g1.xformatter = LONGITUDE_FORMATTER
g1.yformatter = LATITUDE_FORMATTER
cbar = plt.colorbar(f, orientation="horizontal", fraction=0.05, pad=0.07, ticks=[np.log10(0.01), np.log10(0.1),np.log10(0.5), np.log10(1),np.log10(3),np.log10(10),np.log10(50)])
cbar.ax.set_xticklabels(['0.01','0.1','0.5','1','3','10','50'], fontsize=20)
cbar.set_label('Chlorophyll, mg m$^{-3}$', fontsize=20)
plt.title('MODIS [Chl a] mg m$^{-3}$', fontsize=20);

```

basemapと比較すると、これはずっと便利ですが、解像度にはまだいくつかの問題があります。

# 参考資料

https://zhajiman.github.io/post/cartopy_introduction/

https://scitools.org.uk/cartopy/docs/latest/index.html

https://www.net-analysis.com/blog/cartopylayout.html



