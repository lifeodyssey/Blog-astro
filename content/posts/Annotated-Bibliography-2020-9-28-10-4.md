---
title: Annotated Bibliography 2020 9.28-10.4
tags:
  - Remote Sensing
  - Data Fusion
categories: Annotated Bibliography
mathjax: true
abbrlink: e4bb9d98
date: 2020-09-28 12:42:16
copyright:
---

Annotated Bibliography 2020 9.28-10.4

I will try to read three or four papers every week and update my annotation.

Update 2020 9.30: At least two paper.

<!-- more -->

# Downscaling MODIS images with area-to-point regression kriging

Wang, Qunming, et al. "Downscaling MODIS images with area-to-point regression kriging." *Remote Sensing of Environment* 166 (2015): 191-204.

## Method description

### Problem description

$Z_V^l(x_i)$: random variable of pixel V centered at xi (i =1,…,M,where M is the number of pixels) in coarse band l.

$Z_v^k(x_j)$: random variable of pixel v centered at xj (j =1,…,MF^2,where F is the spatial resolution (zoom) ratio between the coarse and fine bands) in fine band k 

The notations v and Vdenote fine and coarse pixels, respectively.

Aim:predict $Z_v^k(x)$ for all fine pixels in all coarse bands. 

Prediction:
$$
Z_v^l(x)=Z_{v1}^l(x)+Z_{v2}^l(x)
$$
$Z_{v1}^l(x)$: regression

$Z_{v2}^l(x)$ : ATPK

### Regression

For coarse band $Z_V^l$ , its corresponding fine band used as covariate is denoted as $Z_v^{lk}$

The regression prediction $Z_{v1}^l(x)$ is calculated as
$$
Z_{v1}^l(x)=a_lZ_v^{lk}(x)+b_l
$$
Key issue is to estimate $a_l$ and $bl$

The relationship in Eq. (2) is assumed to be uni- versal at different spatial resolutions, and the relationship built at coarse spatial resolutions can be applied at fine spatial resolution.

Based on this hypothesis, we get Eq.(3):
$$
Z_V^l(x)=a_lZ_V^{lk}x+b_l
$$
in which $Z_V^{lk}$ is the coarse image produced by upscaling the ancillary fine
band $Z_v^{lk}$ to the same spatial resolution of the l-th coarse band, that is
$$
Z_V^{lk}=h_V^{l}(x)*Z_v^{lk}(x)=\int h_V^l(x-y)Z_v^{lk}(y)dy
$$
where $h_V^l (y)$ is the point spread function (PSF) for the l-th band and ∗ is
the convolution operator

PSF:https://www.youtube.com/watch?v=Tkc_GOCjx7E

ordinary least squares (OLS) is applied for estimate $a_l$ and $b_l$

The ancillary information fromother data, such as elevation data and
field measurement correlated to the observation, can also be favorably incorporated in regression modeling. Given T groups of covariates, $Z_v^t(x)(t =1,…,T)$, the regression prediction $Z_{v1} ^l (x)$ in this more general case then becomes
$$
Z_{v1}^{l}(x)=\sum_{t=0}^Ta_{lt}Z_v^t(x),Z_{v}^{0}(x)=1\forall x
$$

### ATPK

Residuals, denoted as $Z_{V2}^l(x)$,from the regression process, that it,
$$
Z_{V2}^{l}(x)=Z_{V}^{l}-[a_l Z_V^{lk}(x)+b_l]
$$
The results ofregression cannot reproduce the spectral properties of
the observed coarse data. Thus, regression alone is not sufficient for downscaling. The residuals at fine spatial resolution should be compen- sated for the regression prediction to honor the spectral properties. In the proposed ATPRK approach, based on the strong assumption **that the residual is an intrinsically stationary process**, ATPK acts as the second phase to downscale the residuals $Z_{V2}^l (x)$ to fine spatial resolution residuals $Z_{v2} ^l (x)$. 

Based on ATPK, the fine residual $Z_{v2} ^l (x)$ is a linear combination of N coarse residuals of band l
$$
Z_{v2}^{l}(x)=\sum_{i=1}^{N}\lambda_i Z_{V2}^l(x_i),s.t.\sum_{i=1}^N\lambda_i=1
$$
in which $λ_i$ is theweight for the $i$ th residual of the coarse pixel centered at $x_i$.

The $N$ coarse residuals are from the $N$ coarse pixels surrounding the pixel center at $x$,such as $N$=5×5 window of coarse pixels.

As observed from Eq. (7), the ATPK part accounts for the spatial correlation between coarse pixels, which is not utilized in the regression part.

The objective of the ATPK part is to obtain the $N$ weights ${λ_1,…,λ_N}$.
The weights are estimated by minimizing the prediction error variance. The corresponding kriging system is
$$
\begin{bmatrix}
C_{VV}^l(x_1,x_1) & \cdots C_{VV}^l(x_1,x_N)&1\\\\
\vdots&\vdots&\vdots \\\\
C_{VV}^l(x_N,x_1) & \cdots C_{VV}^l(x_N,x_N)&1\\\\
1 \cdots &1 & 0
\end{bmatrix}
\begin{bmatrix}
\lambda_1\\\\
\vdots\\\\
\lambda_N\\\\
\mu
\end{bmatrix}
=
\begin{bmatrix}
C_{vV}^l(x,x_1)\\\\
\vdots\\\\
C_{vV}^l(x,x_N)\\\\
1
\end{bmatrix}
$$
The term $C_{VV} ^l (x_i, x_j)$ is the coarse-to-coarse residual covariance between coarse pixels centered at $x_i$ and $x_j$ in band l, $C_{vV}^ l (x, x_j)$ is the fine-to-coarse residual covariance between fine and coarse pixels centered at $x$ and $x_j$, respectively, and $μ$ is the Lagrange multiplier. The $N$ weights can be calculated according to Eq. (8). For this purpose, the two types of covariance in Eq. (8) need to be obtained in advance

Suppose $s$ is the Euclidean distance between centroids of any two pixels, and $C_{vv} ^l (s)$is the fine-to-fine (called “punctual” in this paper, by assuming each fine pixel as a point) residual covariance between two fine pixels. The fine-to-coarse covariance $C_{vV}l (s) $and coarse-to-coarse covariance $C_{VV}^ l (s)$ are calculated by convoluting $C_{vv}^ l (s)$ with the PSF $h_V ^l (s)$as follows
$$
C_{vV}^l(s)=C_{vv}^l(s)*h_V^l(s)
$$

$$
C_{VV}^l(s)=C_{VV}^l(s)*h_V^l(s)*h_V^l(-s)
$$

In Eq. (10), $−s$ means that the distance from point A within a pixel to point B within another pixel, denoted as $−s$, is opposite to that from point B to point A (i.e., $s$)

Note that the variable in the PSF should be a location, as indicated earlier in Eq. (4). However, s in Eqs. (9) and (10) is originally defined as a distance. Actually, in Eqs. (9) and (10), $s$ is defined for covariance $C_{vv}^l$, which can be recognized as a 2-D image centered at {0,0}, with values in all directions and at multiple lag dis- tances. Thus, $s$ in Eqs. (9) and (10) is essentially a location in the image (e.g., $s$ = {1,1} means a point along the northeast direction with lag 1.414). 

If we assume that the coarse pixel value is the average of the fine pixel values within it, then the PSF is
$$
h_V^l(x)=\begin{cases}\frac{1}{S_v},\quad if\quad x\in V(x)\\
0, \quad otherwise
\end{cases}
$$
where $S_V$ is the size of pixel $V$,and $V(x)$ is the spatial support of pixel $V$ centered at $x$. If the area of pixel $v$ is defined as 1, then $S_V=F^2$.Given the PSF in Eq. (11), the computation of $C_{vV}^l (x, x_j)$and $C_{VV}^l(x_i, x_j)$ are further simplified as 
$$
C_{vV}^l(x,x_j)=\frac{1}{S_V}\int_\limits{u\in V(x_j)}C_{vv}^l(\boldsymbol{x}-\boldsymbol{u})d\boldsymbol{u}=\frac{1}{F^2}\sum_{m=1}^{F^2}C_{vv}^{l}(S_m)
$$

$$
C_{VV}^l(x_i,x_j)=\frac{1}{S_V}\int_\limits{u\in V(x_i)}C_{vV}^l(u,x_j)d\boldsymbol{u}\\\\
=\frac{1}{ {S_V}^2}\int_\limits{u\in V(x_i)}\int_\limits{u^\prime\in V(x_j)}C_{vv}^l(u-u^{\prime})d\boldsymbol{u}d\boldsymbol{u}^{\prime}\\\\
=\frac{1}{F^4}\sum_{m=1}^{F^2}\sum_{m=1}^{F^2}C_{vv}^l(s_{mm^\prime})
$$
In Eq. (12), $s_m$ is the distance between the centroid $x$ of fine pixel $v$ and the centroid of any fine pixel within the coarse pixel $V$ centered at $x_j$,and $s_{mm^\prime}$is the distance between the centroid of any fine pixel within the coarse pixel centered at $x_i$ and the centroid of any fine pixel within the coarse pixel centered at $x_j$. 

The critical problem of kriging weight estimation in ATPK becomes the estimation of punctual residual covariance $C_{vv} ^l (s)$. It is derived by deconvolution (also termed deregularization in geostatistics) of the areal covariance, denoted as $C_V
^l (s)$, of the known coarse residual image $Z_{v2}^l$. Deconvolution aims to estimate the optimal punctual covariance, the regularized covariance of which approximates the known areal covariance.

Given a candidate pool of punctual covariances, each $C_{vv}^l (s)$ is convolved to the regularized areal covariance (denoted as $C_V^{l\_R}(s)$). The optimal punctual covariance is determined as the one with the smallest difference between $C_V^{l\_R}(s)$ and CV l (s). The covariance, which is closely related to the semivariogram (their sum is a constant), can generally be characterized by three parameters, nugget, sill and range. The essence of punctual covariance estimation is the optimal parameter combination estimation. Note that in Eq. (8), the covariance matrix can be replaced by the semivariogram matrix and both give the same kriging weights. Therefore, the covariance modeling, including deconvolution and convolution, is essentially the semivariogram modeling

The candidate pool ofpunctual covariances is generated by referring to the known areal covariance $C_V ^l (s)$. Specifically, for each of the three parameters of$ C_{vv}^ l (s)$, two multipliers are defined empirically to generate an interval for selecting the corresponding optimal parameter of $C_{vv}^l (s)$. In this paper, the interval for punctual sill selection is set to between 1 and 3 times that of the areal sill, while the interval for punctual range selection is set to between 0.5 and 2.5 times that of the areal range. The step is set to 0.1. Regarding the punctual nugget, 21 × 21 = 441 steps of convolution need to be implemented to test each given nugget value. The computational cost increases linearly with the number of nugget candidates. To ease the computational burden, the assumption made in Atkinson et al. (2008) and Pardo-Igúzquiza et al. (2006, 2011) is adopted in this paper: there is zero nugget effect in the punctual covariance. As a result, the optimal punctual parameter combination is selected from the candidate pool containing 441 groups of parameters.

### ATPRK

After both regression and ATPK are completed , their outputs (i.e., $Z_{v1}^ l (x)$and $Z_{v2}^ l (x))$ are combined to produce the final downscaled result, as indicated in Eq. (1).The whole flowchart of the proposed ATPRK approach for downscaling MODIS imagery is shown in Fig. 1. For each coarse band l,ATPRK is performed in turn, and the final result is a fine spatial resolution seven-band MODIS image. The flowchart in Fig. 1 is easy to implement and can be revised by adding other available covariates, as illustrated in Eq. (5).
![image-00020930132948176](/Users/zhenjia/Library/Application Support/typora-user-images/image-00020930132948176.png)

An important advantage of ATPK is the coherence characteristic (Kyriakidis, 2004; Kyriakidis & Yoo, 2005)
$$
\int h_V^l(x-y)Z_{v2}^l(y)dy=Z_{V2}^l(x)
$$
This has been proved theoretically and demonstrated practically.

By compensating the downscaled residuals for regression prediction, we have that
$$
\int h_V^l(x-y)Z_v^l(y)dy=Z_V^l(x)
$$


#### Appendix: Preserve of spectral properties

Without loss of generality, we consider the case of $T$ groups of ancillary variables, that is, the case in Eq. (5). According to Eq. (4), when building the regression model, each group of ancillary data $Z_v^t(x)$ is upscaled to $Z_V ^t(x)$as follows
$$
Z_V^t(x)=\int h_V^l(x-y)Z_v^t(y)dy
$$
In APTK prediction
$$
\int h_V^l(x-y)Z_{v2}^l(y)dy=Z_{V2}^l(x)=Z_V^l(x)-\sum_{t=0}^{T}a_{lt}Z_V^t(x)
$$
Therefore, we have
$$
\int h_V^l(x-y)Z_v^l(y)dy\\\\
=\int h_V^l(x-y)[Z_{v1}^l(y)+Z_{v2}^{l}(y)]dy\\\\
=\int h_V^l(x-y)Z_{v1}^l(y)dy+\int h_V^l(x-y)Z_{v2}^l(y)dy\\\\
=\int h_V^l(x-y)Z_{v1}^l(y)dy+Z_V^l(x)-\sum_{t=0}^{T}a_{lt}Z_V^t(x)\\\\
=\int h_V^l(x-y)\sum _{t=0}^{T}a_{lt}Z_{v}^l(y)dy+Z_V^l(x)-\sum_{t=0}^{T}a_{lt}Z_V^t(x)\\\\
=\sum_{t=0}^{T}a_{lt}\int h_V^l(x-y)Z_{v}^l(y)dy+Z_V^l(x)-\sum_{t=0}^{T}a_{lt}Z_V^t(x)\\\\
=\sum_{t=0}^{T}a_{lt}Z_V^t(x)+Z_V^l(x)-\sum_{t=0}^{T}a_{lt}Z_V^t(x)\\\\
=Z_V^l(x)
$$
This means that when the downscaled result $Z_v^ l (x)$ produced by the
ATPRK approach is upscaled to the original coarse spatial resolution, it is identical to the original observed coarse image$ Z_V^l (x)$. Thus, the spectral properties of the coarse data are precisely preserved by ATPRK. This is a significant advantage of the new method ATPRK in image fusion.

## Summary and Annotation

To be honest, I didn't really understand it in detail, especially the deduction part. But fortunately, the author provided the source code in Matlab(https://github.com/qunmingwang), I can try to rewrite in the python and try to understand it more.

It might be the best method that could preserve the spectral properties(2020.9.30). Of course I need to do more literature review. 

Another strength of this paper is that I now clearly undetstand what do I need to do for writing such kind of articles, including methods comparison(all most all the state-of-art and frequently used method), metrics(the most metrics  I have seen) and classification(the essens of spectral resolution ).

In summary, ATPRK includes two parts: regression and ATPK. The former uses fine spatial resolution information from the fine band and the latter down- scales the residuals from regression that are then added back to the regression prediction.

## Note for ATPRK matlab code

code repository:https://github.com/qunmingwang/Code-for-S2-fusion

This code is for Q. Wang, W. Shi, Z. Li, P. M. Atkinson. Fusion of Sentinel-2 images. Remote Sensing of Environment, 2016, 187: 241�C252.

This paper aims to downscale 20m Sentinel-2 band to 10m. And it compared the result with eight CS and MRA-based approaches.

This code chnage PSF to Gaussian filter. But it is proofed that the coherence characteristics of ATPK and ATPRK is not affected by the specific form of PSF.

### Band Selection

This paper extended the ATPRK using the synthesized and selected band schemes.

For accommodation of fine spatial reso- lution information from multiple fine bands, two schemes were proposed in Selva et al. (2015), that is, the synthesized band scheme and the selected band scheme. 

The selected band scheme selects a fine band from the fine band set for each coarse band, which is determined as the one with the largest correlation with the visited coarse band. 

The synthesized band scheme synthesizes a single fine band from the fine band set (i.e., fine multispectral image), such as averaging all fine multispectral bands (Selva et al. 2015). 

However, the synthesized band scheme based on the simple averaging process fails to consider the relation between the visited coarse band and the four fine bands. In this paper, to fully account for the information in the four finebands, the synthesized band for each coarse band is determined adap- tively as a linear combination ofthe four fine bands.

So this is the why there exist the 'Selected band scheme.m' and 'Synthesized band scheme.m' as the first two file.

After band selection, as the selected band scheme only use panchromatic band and synthesized use multispectral bands, so the function are called 'ATPRK_PANsharpen' and 'ATPRK_MSsharpen', respectively.



# Deriving consistent ocean biological and biogeochemical products from multiple satellite ocean color sensors
Menghua Wang, Lide Jiang, SeungHyun Son, Xiaoming Liu, and Kenneth J. Voss, "Deriving consistent ocean biological and biogeochemical products from multiple satellite ocean color sensors," Opt. Express **28**, 2661-2682 (2020)

 This paper proposed a method that could derive consistent product from different new.

I need to consider sensor response function.

Consistency evaluation by scatter plot and density plot



