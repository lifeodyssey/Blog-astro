---
title: py6s
abbrlink: 514b9dac
date: 2021-12-10 15:04:43
tags:
  - 大气校正
  - Remote Sensing
categories: 学习笔记
mathjax:
copyright:
password:
---

之前一直想方设法跨过去的事情今天终于要开始搞了

<!-- more -->

# 6S介绍

我今天终于才弄明白6到底是怎么做大气校正的

6S的全名是Second Simulation of a Satellite Signal in the Solar Spectrum vector code，所以他不是通过输入的遥感影像来算出来的，而是在选定一个大气环境下，通过辐射传输模拟算出来辐亮度值，然后把观测得到的辐亮度值减去这个算出来的。

而不是像水色里的步骤一样，通过各种假设和经验计算出来辐亮度，再减去辐亮度

然后再来搞清楚一下我要弄的东西

我这里要复现的是High-frequency observation of floating algae from AHI on Himawari-8这篇文章

>The calibration parameters were applied to raw counts
>to produce the top-of-atmosphere radiance ($L_{TOA}$) and reflectance ($R_{TOA}$), and then ($R_{TOA}$) was converted to Rayleigh-corrected reflectance($R_{rc}$) following Hu et al. (2004):
>$$
>R_{TOA}=\pi L_{TOA}/(F_0cos(\theta_0))\\
>R_{rc}=R_{TOA}-R_r
>$$
>where $F_0$ is the extraterrestrial solar irradiance, $θ_0$ is sun zenith angle;
>$R_r$ is Rayleigh reflectance estimated with the 6S model (Vermote et al.,1997) modified for AHI. The modification was simply to apply the
>spectral response function of all bands of AHI into the original 6S
>model. The required inputs to the 6S model, for example, barometric
>pressure, were obtained from ECMWF (https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/).

一个个来

首先是从0级数据到TOA.

在[这个](https://www.eorc.jaxa.jp/ptree/userguide.html)页面看到从1级数据开始就是albedo了，所以还是要从0级数据开始

问题是转换公式是什么呢。

于是我又去看了看他提到的那篇hu2004，但是没找到

回头又看了眼他前面说的那句

>containing all the necessary metadata for calibration and projection parameters in the header section, with the raw binary data stored in the main HSD block.

哦那就不用管了，为了确认我打开[这个页面](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v13.pdf)看了眼

![image-20211216174336134](C:/Users/zhenjia/AppData/Roaming/Typora/typora-user-images/image-20211216174336134.png)

然后还有一个这个

![image-20211217174759219](C:/Users/zhenjia/AppData/Roaming/Typora/typora-user-images/image-20211217174759219.png)

问题来了，这一步算出来的是albedo，但是我要用的是reflectance，这俩一样吗？

我查了[这个](https://gis.stackexchange.com/questions/36726/difference-between-albedo-and-surface-reflectance)回答，没看懂，后来想想，去查单位呗

Radiance的单位是$Wm^{-2}\mu m^{-1}sr^{-1}$,pi和那个cos都是没有单位的

翻到了一篇[古老的文献](https://aslopubs.onlinelibrary.wiley.com/doi/pdf/10.4319/lo.1990.35.8.1657)，里面给出的单位是

$W m^{-2} nm^{-1}$

因为是band averaged（spectral mean）所以有个nm-1正常，和irradiance的单位就差了个这个

所以那个最终的单位是
$$
\frac{Wm^{-2}\mu m^{-1}sr^{-1}}{W m^{-2} nm^{-1}} \\
=sr^{-1}
$$
对我来说差一个角度值，这个角度值只能是Cos(theta0)来的

我就很费解，这个cos(theta0)是从哪儿提供的角度呢

所以这个肯定是不一样。

最后在发邮件问了他们的工作人员之后，终于得到了这个数值。

在[这个](https://www.data.jma.go.jp/mscweb/data/monitoring/gsics/vis/techinfo_visvical.html)页面，算出来的单位也是对的

##　6s

> Rr is Rayleigh reflectance estimated with the 6S mode

在用咕咕噜搜了半天之后，终于找到作者本人写的怎么算的文章了，在[这里](https://blog.rtwilson.com/calculating-rayleigh-reflectance-using-py6s/)。

（百度和bing完全搜不到这篇文章，国内还把bing给禁了，那不就是拿百度给民众喂屎）

以下是对原文的翻译+自我理解

首先我们要理解，什么是瑞利反射。

瑞利反射是单独由大其中的分子造成的反射，而6s并不能单独计算这一项，她能计算的是所有的反射率，是地表+气溶胶+各种分子。

所以想计算瑞利反射，我们需要取消掉其他所有的计算，这意味着我们得到的反射率只包含瑞利反射。

代码是这样的

```python
from Py6S import *

s = SixS()

# Standard altitude settings for the sensor
# and target
s.altitudes.set_sensor_satellite_level()
s.altitudes.set_target_sea_level()

# Wavelength of 0.5nm
s.wavelength = Wavelength(0.5)

# turn off the ground reflectance
s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)

# turn off the aerosol scattering

s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
# turn off the atmospheric absorption by gases
s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.NoGaseousAbsorption)
# then run the simulation and look the output

print(s.outputs.fulltext)

Rc= s.outputs.apparent_reflectance
```



这样，虽然最后计算出来的东西叫apparent reflectance，但是因为我们把瑞利反射外所有的反射都去掉了，所以这个apparent reflectance就是最终的Rayleigh reflectance。

 

除此之外，还有一些有意思的东西。

第一，我们其实不需要把地面的反射率设为0，如果我们把地面反射率设为其他的东西，比如

```python
s.ground_reflectance = GroundReflectance.HomogeneousLambertian(GroundReflectance.GreenVegetation)
```

将会得到一些不一样的数值，这个数值叫做atmospheric intrinsic reflectance，这是直接来自大气的反射率（在本例中只是来自瑞利散射，但在正常情况下，这也包括气溶胶散射）。这可以作为s.output.urgeric_intrinsic_reflectance进行访问。

还有一件事，只是为了表明Py6S中的瑞利反射率的行为方式与我们所了解的物理学知识相一致......我们可以编写一点代码，提取不同波长的瑞利反射率并绘制成图--我们希望有一条指数递减的曲线，显示低波长的高瑞利反射率，反之亦然。

```python
from Py6S import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

s = SixS()

s.altitudes.set_sensor_satellite_level()
s.altitudes.set_target_sea_level()
s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.NoGaseousAbsorption)

wavelengths = np.arange(0.3, 1.0, 0.05)
results = []

for wv in wavelengths:
    s.wavelength = Wavelength(wv)
    s.run()

    results.append({'wavelength': wv,
                   'rayleigh_refl': s.outputs.atmospheric_intrinsic_reflectance})

results = pd.DataFrame(results)

results.plot(x='wavelength', y='rayleigh_refl', style='x-', label='Rayleigh Reflectance', grid=True)
plt.xlabel('Wavelength ($\mu m$)')
plt.ylabel('Rayleigh Reflectance (no units)')
```

这个图是

![file](https://blog.rtwilson.com/wp-content/uploads/2019/09/image-1569879583877.png)

在这一大块代码中没有什么特别革命性的东西--我们只是结合了我之前演示的代码，然后在各种波长中循环，为每个波长运行模型。

我们存储模型结果的方式值得简单解释一下，因为这是我经常使用的一种模式。每次运行模型，都会有一个新的dict被添加到一个列表中--这个dict有我们感兴趣的各种参数（在这里只是波长）和我们感兴趣的各种结果（在这里只是Rayleigh反射率）的条目。循环结束后，我们可以简单地将这个dict列表传递给pd.DataFrame()，然后得到一个漂亮的pandas DataFrame - 准备好显示、绘制或进一步分析。

其他的一些设置

瑞利反射率将根据视角和太阳角的不同而变化。你可以通过s.geometry = Geometry.User()手动设置，然后设置s.geometry.solar_z, solar_a等。或者你可以通过使用Geometry.User.from_time_and_location()来半自动地设置，它将根据位置和时间自动设置太阳角，并允许你手动设置视图角度。

至于大气压力等等，它实际上是作为设置海拔高度的一部分而配置的。你可以在高度方面设置高度，也可以在压力方面设置高度--见https://py6s.readthedocs.io/en/latest/params.html#Py6S.Altitudes.set_target_pressure，如果你有任何进一步的问题，请随时询问。

