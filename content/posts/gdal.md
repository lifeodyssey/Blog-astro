---
title: gdal
tags:
  - gdal
  - remote sensing
  - image processing
categories: 学习笔记
abbrlink: f08904cf
date: 2021-12-22 15:17:31
mathjax:
copyright:
password:
---

之前就想了各种各样的方法来跳过GDAL，没想到最后工作了还是要用这个

<!-- more -->

#　安装



# 直接装

```bash
$ conda install -c conda-forge gdal
```

反正我是这么装好的

# 下载再装

下载proj-6.2.0.tar.gz在https://proj.org/download.html

然后

```bash
$　tar -xf proj-6.2.0.tar.gz 
$ yum install sqlite-devel
$ cd proj
$ ./configure
$ make
$ make check
$ make install
```

接着下载gdal

```bash
$ wget -c http://download.osgeo.org/gdal/3.1.4/gdal-3.1.4.tar.gz
$　tar -zxvf gdal-3.1.4.tar.gz
$ cd gdal-3.1.4
$ sudo yum install -y gcc 
$ make subversion gcc-c++ sqlite-devel libxml2-devel python-devel numpy swig expat-devel libcurl-devel
$ ./configure
$ make
$ sudo make install
$ pip install GDAL==3.1.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## 读取



## 剪裁

##　重投影

## 存储

参考的[这个](https://here.isnew.info/how-to-save-a-numpy-array-as-a-geotiff-file-using-gdal.html)

```python
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

def read_geotiff(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds
# retrun ds是因为后面需要一样的投影和坐标系
def write_geotiff(filename, arr, in_ds):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32
	# 确定是用arr本身的数据类型还是其他的数据类型
    driver = gdal.GetDriverByName("GTiff")
    # 确定输出数据类型
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    # 确定一些输出数据的参数，包括文件名，column和height，多少个波段(那个1)，什么数据类型
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    # 确定数据的投影和坐标系，这里他用的是原本的那个
    # 除此之外，还可以使用绝对的坐标系
    # GeoTransform 的形式为 (486892.5, 15.0, 0.0, 4105507.5, 0.0, -15.0)
    # 六个参数分别为 左上角x坐标， 水平分辨率，旋转参数， 左上角y坐标，旋转参数，竖直分辨率
    # 一般旋转参数都设为0
    # SetProjection则可以是OGC WKT或者PROJ.4格式的字符串 可以从https://cfconventions.org/wkt-proj-4.html查到
    
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)

nlcd01_arr, nlcd01_ds = read_geotiff("nlcd2001_clipped.tif")
nlcd16_arr, nlcd16_ds = read_geotiff("nlcd2016_clipped.tif")

nlcd_changed = np.where(nlcd01_arr != nlcd16_arr, 1, 0)

write_geotiff("nlcd_changed.tif", nlcd_changed, nlcd01_ds)

plt.subplot(311)
plt.imshow(nlcd01_arr)

plt.subplot(312)
plt.imshow(nlcd16_arr)

plt.subplot(313)
plt.imshow(nlcd_changed)

plt.show()
```

