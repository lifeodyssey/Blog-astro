---
title: cartopy
copyright: true
tags:
  - Research
  - python
categories: 学习笔记
mathjax: true
abbrlink: c6b8edee
date: 2020-08-01 19:57:05
---

Basemap在2020年底停止维护，取而代之的是cartopy，在这里写一下一些学习笔记。

先说结论，截止到2020年8月1日,0.18版本仍然不能完全取代basemap，尤其是近岸数据分辨率的问题，但是已经展现出优势了，自己要在之后多多尝试使用。

<!-- more -->

# 代码范例

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

相比basemap这个方便很多，但是分辨率还是有一点问题。

# 一些参考资料

https://zhajiman.github.io/post/cartopy_introduction/

https://scitools.org.uk/cartopy/docs/latest/index.html

https://www.net-analysis.com/blog/cartopylayout.html