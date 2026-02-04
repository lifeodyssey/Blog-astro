---
title: MODIS OC data batch download
copyright: true
tags:
  - Research
  - Ocean Color
  - Oceanography
categories: 学习笔记
mathjax: true
abbrlink: 8636bca2
date: 2020-02-20 14:05:08
lang: en
---
This blog introduce a MODIS(also suitable for other available satellites) Ocean Color data batch download method without wget(since it doesn’t work on my laptop).

<!-- more -->

## Requirement

1.Earthdata account

2.cygwim(https://www.cygwin.com/)

## Step by Step

1.Login to Earthdata. Then enter [https://search.earthdata.nasa.gov/search?m=0!-0.0703125!2!1!0!0%2C2&fdc=Ocean%20Biology%20Distributed%20Active%20Archive%20Center%20(OB.DAAC)&ac=true](https://search.earthdata.nasa.gov/search?m=0!-0.0703125!2!1!0!0%2C2&fdc=Ocean Biology Distributed Active Archive Center (OB.DAAC)&ac=true)

2.Select desired satellite in the option ’Instruments’

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20200220141643.png)

3.Select desired region by spatial polygon, rectangular or coordinate.

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20200220141833.png)

Then select desired product 

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20200220142022.png)

4.Screen the data by granule filters

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20200220142153.png)

5.Click download all, then choose data access method and click download data in the end

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/20200220142309.png)

6.Click download access script

![](I:\blog\source\_posts\MODIS-L2-OC-batch-download\20200220142411.png)

download the script

![1582176279212](I:\blog\source\_posts\MODIS-L2-OC-batch-download\1582176279212.png)

open cygwin in the same download folder

enter:

`chmod 777 download.sh`

`./download.sh`

Then enter your password. Waiting for the result.

