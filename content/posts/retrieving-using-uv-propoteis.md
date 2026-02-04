---
title: retrieving Scdm using uv propoteis
tags:
  - Remote Sensing
  - CDOM
  - Inversion
  - Ocean Color
categories: Annotated Bibliography
mathjax: true
abbrlink: 750388a9
date: 2021-01-26 19:22:12
copyright:
lang: en
---
Spectral slopes of the absorption coefficient of colored dissolved and detrital material inverted from UV-visible remote sensing reflectance(Wei et al., 2016)

A new algorithm to retrieve chromophoric dissolved organic matter (CDOM) absorption spectra in the UV from ocean color(Cao and Miller, 2015)

A model for remote estimation of ultraviolet absorption by chromophoric dissolved organic matter based on the global distribution of spectral slope(Swan et al.,2016)

Pan-Arctic distributions of continental runoff in the Arctic Ocean(Fiche et al., 2013)

Algorithm development and validation ofCDOMproperties for estuarine and continental shelfwaters along the northeastern U.S. coast(Mannino er al., 2014)

Retrieval of phytoplankton and colored detrital matter absorption coefficients with remote sensing reflectance in an ultraviolet band(Wei and Lee, 2015)

<!-- more -->

# $S_{cdm}$



There are not so much analytical or semi-analytical algorithm to estimate $S_{cdm}$(Spectral slope of cdom and detriutus absorption)

One thing need to notice is that the $S_{cdm}$ is not strictly power-law.

Wei et al just stress the importanct of uv band. The approach is HOPE.

Wei and Lee, 2015
$$
S=0.00854+\frac{0.005055}{0.2236+r_{rs}(380)/r_{rs}(440)}
$$
It is same with lee 2002 using lee 1998 data. Not so strong relationship

But it might could be used.

Mannino et al., 2014, just cdom and the slope of cdom
$$
ln(S)=B_0+B_1X_1+B_2X_2+B_3X_3+\cdots+B_nX_n
$$
$X_{i..n}=ln[R_{rs}(\lambda_1...\lambda_n)]$

emprical

Fiche et al 2013

Also cdom slope

![image-00030126195045610](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030126195045610.png)

Swan et al., 2013

Just cdom

![image-00030126195239868](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030126195239868.png)

Cao and William., 2014

Just CDOM

PCA+MLR

I need to read HOPE

# CDOM concentration

