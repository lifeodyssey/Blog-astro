---
title: absorption measurement process
tags:
  - Experiment Protocols
  - Absorption Measurement
  - Ocean Color
categories: 学习笔记
mathjax: true
abbrlink: 49ca7bc7
copyright: true
date: 2021-02-24 10:44:21
---

This is a summary of how to process absorption measurement data.

The measurement is basically based on the QFT-T mode. More details could be found at [IOCCG protocols volume 1]( https://ioccg.org/what-we-do/ioccg-publications/ocean-optics-protocols-satellite-ocean-colour-sensor-validation/ ) 

<!-- more -->

# CDOM

![image-00030224105720968](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030224105720968.png)

This is the excel sheet we used for CDOM measurement

Each time we measure two blank and two samples. 

The $a_{CDOM}$ is calcualted follow:
$$
a_{CDOM}=\frac{2.303}{l}[[OD_s(\lambda)-OD_{bs}(\lambda)]-OD_{null}]
$$
$l$ is the path length of the quartz cell. In MPS-2400, the $l=10cm(0.1m)$.  2.303 is the factor to convert $log_{10}$ to $log_{e}$

$OD_s(\lambda)$ is the optical density of the filtered water sample, $OD_{bs}(\lambda)$ is the optical density of purified water, and $OD_{null}(\lambda)$ is the apparent residual optical density at a long visible or near infrared wavelength where absorption by dissolved materials is assumed to be zero. In the previous measurement, we use the 700 nm as null wavelength.

So this excel file is quite clear.

# ap ad aph

The optical density of total particles retained on the filter ($OD_p(λ)$) were kept below 0.3 by adjusting the filtration volume. Subsequently, $OD_p(λ)$ was measured between 350 and 750 nm at 1 nm intervals, using a dual beam multi-purpose spectrophotometer (MPS-2400, Shimadzu Inc.). A blank filter soaked with 0.2 µm-filtered seawater (FSW) was used as the reference. To correct the path length amplification effect caused by multiple scattering in the glass fibre filter, the following equation was utilized according to Cleveland and Weidemann (1993):
$$
OD_s(\lambda)=0.378OD_p(\lambda)+0.523D_p(\lambda)^2
$$
where $OD_s(λ)$ is the optical density of particulate matter in suspension. The absorption coefficient of the total particles ($ap(λ)$) was calculated using the following equation:
$$
a_p(\lambda)=2.303OD_s(\lambda)S/V
$$
where 2.303 is the factor used to convert $log_{10}$ to $log_e$ , S is the filter clearance area ($m^2$ ), and V is the filtered volume ($m^3$ ). The ratio of S to V approximates the geometrical light pass length.

After measurements of the optical density of the total particles, the filters were soaked in methanol for at least 24 h to extract phytoplankton pigments, and then rinsed with FSW. The absorbance of a decolourized filter was re-measured using the same method to obtain the optical density of the non-phytoplankton particles (Kishino et al., 1985). Similarly, the absorption coefficient of non-alga particles ($a_{nah}(λ)$) was determined using Eq. (2) and (3). $a_{ph}(λ)$ was then obtained from the following equation:
$$
a_{ph}(\lambda)=a_{p}(\lambda)-a_{nap}(\lambda)
$$

![image-00030225145047326](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030225145047326.png)

This the the excel sheet for ap. Blank 1&Blank2 are measrued before ap, Blank 3&Blank4 are measrued after ap