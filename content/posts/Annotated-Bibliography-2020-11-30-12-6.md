---
title: Annotated Bibliography 2020 11.30-12.6
tags:
  - Ocean Color
  - Inherent Optical Properties
  - Ocean Optics
  - Research Basis
categories: Annotated Bibliography
mathjax: true
abbrlink: adc0cb59
date: 2020-11-30 11:56:28
copyright:
---

重新加油

这周可以说是IOP专题了

<!-- more -->

# Variations of light absorption by suspended particles with chlorophyll a concentration in oceanic (case 1) waters: Analysis and implications for bio-optical models

This is a fairly old paper.

The importance of this paper for my research is these figures.

![image-00021201104954518](/Users/zhenjia/Library/Application Support/typora-user-images/image-00021201104954518.png)

Although the data is obtained from oceanic(case 1) water, but we can see that the range of chlorophyll is relative high. So the parameteration of $a_{ph}$ might also could be used for case 2 water.

This is the parameteration for $a_{ph}$

![image-00021201105305213](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00021201105305213.png)

Compared with former Bricaud 1995, the r2 is better.

![image-00021201105504185](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00021201105504185.png)

The author put the data here https://doi.pangaea.de/10.1594/PANGAEA.739879

But I use the data from herehttps://github.com/BrandonSmithJ/MDN/blob/abad4338f495f801b88a32c141b6639e7b7989ad/IOP/bricaud_1998_aph.txt

But when I tried this parameteratzation, this actually also produce some strange shape.

![Bricaud1998](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/Bricaud1998.png)

I think this is not only the problem of QAA, because when I use in-situ chia, the aph shape is also strange.

I think the reparameterazation is needed.

# Inversion of In Situ Light Absorption and Attenuation Measurements to Estimate Constituent Concentrations in Optically Complex Shelf Seas

Packaging effect:

It is well known that the chlorophyll-specific phytoplankton absorption coefficient decreases with increasing Chl concentration due to the pigment packaging effect (Bricaud et al., 1995).

This paper suggest that 'linear regression can be performed for the ranges of chlorophyll concentration values that are relevant for shelf seas as the effect of pigment packaging will be relatively limited for these values'

It just assumes that the $a^*_{ph}$ is constant.

So it could fix the value...

But one thing for me is that may be I can try water classification.

There are a lot of things I think.

By now, especially before AWOC. I do not want to solving these problems.

# Light scattering and chlorophyll concentration in case 1 waters: A reexamination

## Intro and MM

 This analysis, restricted to case 1 waters, aims at reassessing a previous nonlinear relationship established between the particle scattering coefficient, $b_p$ (very close
to the particle attenuation coefficient, $c_p$), and the chlorophyll concentration, [Chl]. This paper also suggested a modified criterion for turbid case 2 water.

In Gordon and Morel 1983,
$$
b_{p}(550)=A[Chl]^{0.62}
$$
A, which is on average 0.30 within the upper oceanic layer, may vary between 0.12 and 0.45 to account for the lowest and highest particle scattering coefficients found at various depths in waters sat- isfying the criterion for belonging to case 1 waters 

The importance of Gordon and Morel 1983 is the finding of this nolinear characteristics. But is lacks tightness expressed by the wide possible variation of A.

The Chl is measured by HPLC. The whole dataset has been seperated into several subsets for different purpose. For my research, I just need to focus on the surface, which is the $N_{sat}/Z_{90}$

![image-00021130141721276](/Users/zhenjia/Library/Application Support/typora-user-images/image-00021130141721276.png)

The author also use Bricaud 1995 to model phytoplankton absorption in order to study the contribution of absorption to attenuation coefficient.
$$
a_{ph}(660)=0.012[Chl]^{0.878}
$$

$$
a_{p}(660)=0.014[Chl]^{0.817}
$$

But the coefficient of determination($r^2=0.27$) is relative low.

They found the contribution is negligible. $c_p$ could be considered as same as $b_p$

## Result and discussion

For the near surface layer,
$$
c_{p}(660)=0.347[Chl]^{0.766}(r^2=0.89, N_{sat}=435)
$$

# Particulate backscattering ratio at LEO 15 and its use to study particle composition and distribution

## Intro

