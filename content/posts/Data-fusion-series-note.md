---
title: Data fusion series note 1
copyright: true
tags:
  - paper reading
  - Research Basis
  - Data Fusion
categories: 学习笔记
mathjax: true
abbrlink: bd7e58b0
date: 2020-07-15 22:52:41
---

There are five main data fusion approaches according to 'Spatio-temporal fusion for remote sensing data: an overview and new benchmark'. Here I will read the STARFM.

<!-- more -->

# Weight function-based methods: STARFM

The spatial and temporal adaptive reflectance fusion model (STARFM) is the first weight function-based STF method developed in the literature. This method first assumes that all the pixels in the coarse images are pure. It uses a weighted strategy to add the reflectance changes between two coarse images to the prior fine image so as to predict the target image. STARFM has been shown to be able to capture phenological changes. However, its performance in highly heterogeneous landscapes and in the task of capturing land-cover changes is limited.

## Data Requirement

(MODIS and Landsat)Their orbital parameters are equal, and as such the viewing (near-nadir) and solar geometries are close to those of the corresponding Landsat acquisition.

## Performance

The STARFM has been tested over forested areas, cropland regions, and heterogeneous mixtures of crop and forest. Results show that the STARFM can capture phenology changes precisely, although the accuracy depends on the characteristic patch size of the landscape.

## Theoretical Basis

Assumptions

1. Neglecting geolocation errors and differences in atmospheric correction
2. MODIS surface reflectance $M(x_i,y_j,t_k)$ has been previously georeferenced and super sampled to the resolution and bounds of the Landsat surface reflectance image $L(x_i,y_j,t_k) $and thus shares the same image size, pixel size, and coordinate system. 
3. if MODIS and Landsat surface reflectance are equal at a given time, then these values should be equal for the prediction date.
4. If the MODIS surface reflectance is constant over time, then the Landsat surface reflectance should not change as well.



### Basic Equations

Assumption 1

A heterogeneous coarse-resolution pixel at date *t* and surface reflectance ($C_t$) can be aggregated from finer resolution homogeneous pixels of surface reflectance $F^i_t$ and percentage coverage $A^i_t$ according to
$$
C_t=\sum(F^i_t*A^i_t)\tag{1}
$$

where $i$ refers to the spatial index (location) of the fine- resolution pixel.

The key to finding an approximate solution is to find spectrally similar homogeneous neighboring pixels.

For a homogenous pixel at a coarser MODIS resolution, the surface reflectance measured by Landsat data can be ex- pressed as
$$
L(x_i,y_i,t_k)=M(x_i,y_i,t_i)+\varepsilon_k \tag{2}
$$
where $(x_i,y_j)$ is a given pixel location for both Landsat and MODIS images, $t_k$ is the acquisition date for both MODIS and Landsat data, and $ε_k$ represents the difference between observed MODIS and Landsat surface reflectance (caused by differing bandwidth and solar geometry).

Assumption 2

Suppose we have n pairs input of $L(x_i,y_j,t_k) $and $M(x_i,y_j,t_k)$ and each pair is acquired on the same date, where $k ∈ [1,n]$. The daily MODIS surface reflectance $M(x_i,y_j,t_0)$ at date $t_0$ is also a known value among inputs, then the predicted Landsat surface reflectance at date $t_0$ is
$$
L(x_i,y_j,t_0)= M(x_i,y_j,t_0)+ ε_0. \tag{3}
$$
Suppose the ground coverage type and system errors at pixel$(x_i,y_j)$ does not change over prediction date $t_0$ and the date $t_k$, we will have $ε_0 = ε_k$ and thus
$$
L(x_i,y_j,t_0)= M(x_i,y_j,t_0)+ L(x_i,y_j,t_k) −M(x_i,y_j,t_k).\tag{4}
$$
Such ideal situation cannot be satisfied from MODIS  and Landsat observations. Their relationships are complicated by several factors:

1. MODIS observation is not a homogeneous pixel and may include mixed land-cover types when considered at Landsat spatial resolution
2. Land cover may change from one type to another type during the prediction period
3. Land-cover status (phenology) and solar geometry bidirectional reflectance distribution function (BRDF) changes will alter the reflectance from prediction date $t_0$ to date $t_k$.

By introducing additional information from neighboring pixels, we compute the surface reflectance for the central pixel at date t0 with a weighting function
$$
L (x_{w/2},y_{w/2},t_0) = \sum_{i=1}^{w} \sum_{j=1}^{w} \sum_{k=1}^{w}
W_{ijk}×(M(x_i,y_j,t_0)+ L(x_i,y_j,t_k) −M(x_i,y_j,t_k))\tag{5}
$$
where $w$ is the searching window size and $(x_{w/2},y_{w/2})$ is the central pixel of this moving window. 

To ensure that the right information from neighbor pixels is used, only spectrally similar (i.e., from the same spectral class) and cloud-free pixels from Landsat surface reflectance within the moving window are used to compute the reflectance.

The weight $W_{ijk}$ determines how much each neighboring
pixel contributes to the estimated reflectance of the central pixel. It is very important and is determined by three measures as follows.

1. Spectral difference between MODIS and ETM+ data at a given location is
   $$
   S_{ijk} = |L(x_i,y_j,t_k) −M(x_i,y_j,t_k)| .\tag{6}
   $$
   A smaller value of $S_{ijk}$ implies that the fine spatial resolution pixel has closer spectral features to the averaged surrounding pixels; thus, the change at fine resolution should be comparable to that of the averaged surrounding pixels. Therefore, the pixel’s reflectance should be assigned a higher weight in (5).

   A3

2. Temporal difference between the input and the predicted MODIS data is
   $$
   T_{ijk} = |M(x_i,y_j,t_k) −M(x_i,y_j,t_0)|\tag{7}
   $$
   This metric measures changes occurring between the prediction and the acquisition dates. A smaller $T_{ijk}$ means less vegetation change between time $t_k$ and $t_0$; thus, the pixel should be assigned a higher weight.

   A4

   if changes are too subtle to be detected by the MODIS observation, this algorithm will not be able to predict any change when synthesizing the fine resolution imagery. Also, there may be situations where the STARFM algorithm cannot detect changes when two contradicting changes occur within a coarse-resolution pixel simultaneously and compensate for each other.

3. Location distance between central pixel $(x_{w/2},y_{w/2})$ and candidate pixel $(x_i,y_j)$ at date $t_k$ is
   $$
   d_{ijk} = \sqrt{(x_{w/2} − x_i )^2 + (y_{w/2} − y_j)^2}
   $$
   The spatial similarity is normally better for a closer pixel; thus, the closer candidate should be assigned a higher weight.

## Implementation Considerations

### How to weight spatial information

#### Spectrally Similar Neighbor Pixels

The spectral similarity ensures that the correct spectral information is used from fine-resolution neighboring pixels: Unsupervised classification and using thresholds in surface reflectance directly. STARFM use the second approach. 

"the purpose of the search process is to find pixels within the local moving window that are spectrally similar to the central pixel. Each central pixel becomes the center of the class, and the rules used to determine spectral similarity become local rules and thus vary from pixel to pixel. In contrast to the traditional classification, which applies the same classification rules over the whole region, our search process (second approach) will not be able to produce a unique classification map over the study area."

#### Combined Weighting Function

Based one these assumptions:

1. coarse-resolution homogeneous pixels provide identical temporal changes as fine-resolution observations from the same spectral class
2. observations with less change from the prediction date provide better information for the prediction date
3. more proximal neighboring pixels normally provide better information for prediction.

The final step is to combine these independent factors to create an ideal weight function that blends both temporal and spatial information

First, convert the actual distance to a relative distance through the function
$$
D_{ijk} =1.0+ d_{ijk}/A\tag{9}
$$
where $A$ is a constant that defines the relative importance of the spatial distance to the spectral and temporal distance.

The relative distance $D_ijk$ within searching area “$w$” changes from 1to $[1 + (1/
\sqrt2) ∗ (w/A)]$. A smaller value of $A$ gives a larger dynamic range of $D_{ijk}$.

The combined spectral, temporal, and spatial distance can be computed with
$$
C_{ijk} = S_{ijk} ∗ T_{ijk} ∗ D_{ijk}\tag{10}
$$
or in a logistic formula to make it less sensitive to the spectral differences
$$
C_{ijk} =ln(S_{ijk} ∗ B +1) ∗ ln(T_{ijk} ∗ B +1) ∗ D_{ijk}\tag{11}
$$
where $B$ is a scale factor (equal to 10 000 when using MODIS or LEDAPS reflectance products, which linearly scale reflectance from 0 to 10 000).

We use a normalized reverse distance as the weight function

$$
W_{ijk} =(1/C_{ijk}) /\sum_{i=1}^{w}\sum_{j=1}^{w}\sum_{k=1}^{n}(1/C_{ijk}).\tag{12}
$$

If the MODIS surface reflectance does not change, we have
$M(x_i,y_j,t_k)= M(x_i,y_j,t_0)$, then $T_{ijk} =0$ and $C_{ijk} =0$, and weight $W_{ijk}$ is set to the maximum value. The predicted surface reflectance for central pixel of the moving window is then

$$
L (x_{w/2},y_{w/2},t_0) = M(x_i,y_j,t_0).
$$

This satisfies our other basic assumption: if MODIS and Landsat surface reflectance are equal at date $t_k$, then they should be equal at date$ t_0$

#### Sample Filtering

Additional filtering processes will then be applied to the candidates to remove poor- quality observations. 

1. All poor-quality data are excluded from candidates according to the QA layer in the Landsat and MODIS surface reflectance products

2. Neighbor pixels are filtered out if they cannot provide better spectral and spatial information than the central pixel of the moving window

   A good candidate should satisfy the following condition:
   $$
   S_{ijk} < max (|L(x_{w/2},y_{w/2},t_k) −M(x_{w/2},y_{w/2},t_k)|)\tag{13}
   $$
   and
   $$
   T_{ijk} < max(|M(x_{w/2},y_{w/2},t_k) −M(x_{w/2},y_{w/2},t_0)|) \tag{14}
   $$
   

Suppose we know that the uncertainties from Landsat and MODIS surface reflectance are $\sigma_l$ and $\sigma_m$, respectively. All surface reflectance measurements are independent. The uncertainty for the spectral difference (6) between MODIS and ETM+ is
$$
\sigma_{lm} = \sqrt{\sigma_l^2+\sigma_m^2}
$$
The uncertainty for temporal difference (7) between two MODIS inputs is
$$
\sigma_{mm} = \sqrt{\sigma_m^2+\sigma_m^2}=\sqrt{2}*\sigma_m
$$
Considering the uncertainties in the candidate selection, (12)can be revised as

$$
S_{ijk}< max(|{L(x_{w/2},y_{w/2},t_k)−M(x_{w/2},y_{w/2},t_k)}
+σ_{lm}\tag{15}
$$

and(13)can be revised as

$$
T_{ijk} <max(|M(x_{w/2},y_{w/2},t_{k}−M(x_{w/2},y_{w/2},t_0) + \sigma_{mm}\tag{16}
$$

## Following methods

http://www.chen-lab.club/?page_id=11432

ESTARFM,IFSDAF,cuFSDAD

# FSDAF

最后选用的是这个算法，也懒得去重新写ISFDAF的代码了。有GPU的话可以使用cuFSDAF。更好的选择是IFSDAF或者FSDAF 2.0 但是都是IDL的代码 我不会IDL.jpg

代码在https://xiaolinzhu.weebly.com/open-source-code.html可以找到

流程

![image-20220207140829822](C:/Users/zhenjia/AppData/Roaming/Typora/typora-user-images/image-20220207140829822.png)



简单看下这个代码吧，看看需要改哪里的参数之类的

基本都在FSDAF.py里

没啥需要改的参数 就是用起来挺麻烦的.

关于参数的设置

软件中涉及到的融合参数没有标准答案。最近一项研究中（Zhou et al., 2021）中的设置可作为初始尝试：

1) Number of Classes 表示影像中土地覆盖类型数量，一般设置为5-7（STARFM可设置更多数量，如25）；

2) Number of Similar Pixels 表示算法中搜索周边相似像元的数量，与粗分辨率与细分辨率的比值为相关（假设为R），一般设置为Ｒ*1.5再向上取整。如MODIS-Landsat分辨率比值为8，则该值取12

3) Windows size(half width) 表示相似像元的搜索窗口，与粗分辨率与细分辨率的比值为相关（假设为R），一般设置为（R*1.5）/2再向上取整. 如MODIS-Landsat分辨率比值为8，则该值取6
4) Min/Max value表示融合数据的最大值和最小值，根据你的数据格式取值，如0-1或者0-255等
Zhou J., Chen J., Chen X.*, Zhu X., Qiu, Y., Song, H., Rao, Y., Zhang C., Cao X., Cui X. (2021).Sensitivity of six typical spatiotemporal fusion methods to different influential factors:A comparative study for a normalized difference vegetation index time series reconstruction. Remote Sensing of Environment .252, 112130. doi.org: 10.1016/j.rse.2020.112130

摁跑了之后发现没啥结果，还是要认真看步骤找一下结果

## 1. Fine Resolution Classification

第一步是先把高分辨率的做一个分类，这里用的是ISODATA。

做完之后可以得到在一个粗分辨率像素中不同类的比例
$$
f_{c}(x_i,y_i)=N_C(x_i,y_i)/m
$$
Nc是在粗分辨率xi,yi处属于c类的像素数量

## 2.计算每一类的时间变化

