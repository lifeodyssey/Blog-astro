---
title: probabilistic method in machine learning
tags:
  - 机器学习
  - Inversion
  - Deep Learning
  - tensorflow
categories:
  - 学习笔记
abbrlink: 62469dc9
date: 2022-02-12 14:24:22
mathjax:
copyright:
---

机器学习中的统计方法，主要是跟着mixture density network那篇下来的

<!-- more -->

在MDN的那篇里面主要使用的是tfp.distributions.MixtureSameFamily(prob, dist),

mix.log_prob(y)

MixtureSameFamily，一个categorical,

```python
def loss(self, y, output):
    prior, mu, scale = self._parse_outputs(output) 
    #获取输出结果的三个参数
    dist  = getattr(tfp.distributions, self.distribution)(mu, scale)
    prob  = tfp.distributions.Categorical(probs=prior)
    mix   = tfp.distributions.MixtureSameFamily(prob, dist)
    likelihood = mix.log_prob(y)
    return tf.reduce_mean(-likelihood) + tf.add_n([0.] + self.model.losses)
```

就是这个函数

来细致看里面调用的东西

# 第一句

```python
 dist  = getattr(tfp.distributions, self.distribution)(mu, scale)
```

getattr获得对象的属性值

```python
getattr(object, name[, default])
```

- oject -- 对象。
- name -- 字符串，对象属性。
- default -- 默认返回值，如果不提供该参数，在没有对应属性时，将触发 AttributeError。

但是这里用的是利用getattr来调用函数的办法

self.distribution等于 'MultivariateNormalTriL'这句话实际上是 在tfp.distributions里调用了一个 'MultivariateNormalTriL'的函数实例，然后把(mu, scale)这两个参数传给这个函数

这个函数是

````python
class MultivariateNormalTriL(
    mvn_linear_operator.MultivariateNormalLinearOperator):
  """The multivariate normal distribution on `R^k`.
  The Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
  `scale` matrix; `covariance = scale @ scale.T` where `@` denotes
  matrix-multiplication.
  #### Mathematical Details
  The probability density function (pdf) is,
  ```none
  pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z,
  y = inv(scale) @ (x - loc),
  Z = (2 pi)**(0.5 k) |det(scale)|,
  where:
  * `loc` is a vector in `R^k`,
  * `scale` is a matrix in `R^{k x k}`, `covariance = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||y||**2` denotes the squared Euclidean norm of `y`.
    A (non-batch) `scale` matrix is:
  ```none
  scale = scale_tril
```
  where `scale_tril` is lower-triangular `k x k` matrix with non-zero diagonal,
  i.e., `tf.diag_part(scale_tril) != 0`.
  Additional leading dimensions (if any) will index batches.
  The MultivariateNormal distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,
  ```none
  X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
  Y = scale @ X + loc
  ```
  Trainable (batch) lower-triangular matrices can be created with
  `tfp.distributions.matrix_diag_transform()` and/or
  `tfp.math.fill_triangular()
````


简单理解就是传入了方差和均值两个参数，来计算一个多变量的正态分布

```python

  tfd = tfp.distributions
  # Initialize a single 3-variate Gaussian.
  mu = [1., 2, 3]
  cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
  scale = tf.linalg.cholesky(cov)
  # ==> [[ 0.6,  0. ,  0. ],
  #      [ 0.2,  0.5,  0. ],
  #      [ 0.1, -0.3,  0.4]])
  mvn = tfd.MultivariateNormalTriL(
      loc=mu,
      scale_tril=scale)
  mvn.mean()
  # ==> [1., 2, 3]
  # Covariance agrees with cholesky(cov) parameterization.
  mvn.covariance()
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]
  # Compute the pdf of an observation in `R^3` ; return a scalar.
  mvn.prob([-1., 0, 1])  # shape: []
  # Initialize a 2-batch of 3-variate Gaussians.
  mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
  tril = ...  # shape: [2, 3, 3], lower triangular, non-zero diagonal.
  mvn = tfd.MultivariateNormalTriL(
      loc=mu,
      scale_tril=tril)
  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x)           # shape: [2]
  # Instantiate a "learnable" MVN.
  dims = 4
  mvn = tfd.MultivariateNormalTriL(
      loc=tf.Variable(tf.zeros([dims], dtype=tf.float32), name="mu"),
      scale_tril=tfp.util.TransformedVariable(
          tf.eye(dims, dtype=tf.float32),
          tfp.bijectors.FillScaleTriL(),
          name="raw_scale_tril")
```

# 第二句

在获取了这个分布之后

```python
prob  = tfp.distributions.Categorical(probs=prior)
```

```python
class Categorical(distribution.AutoCompositeTensorDistribution):
  """Categorical distribution over integers.
  The Categorical distribution is parameterized by either probabilities or
  log-probabilities of a set of `K` classes. It is defined over the integers
  `{0, 1, ..., K-1}`.
  The Categorical distribution is closely related to the `OneHotCategorical` and
  `Multinomial` distributions.  The Categorical distribution can be intuited as
  generating samples according to `argmax{ OneHotCategorical(probs) }` itself
  being identical to `argmax{ Multinomial(probs, total_count=1) }`.
  #### Mathematical Details
  The probability mass function (pmf) is,
  ```none
  pmf(k; pi) = prod_j pi_j**[k == j]

Pitfalls
  The number of classes, `K`, must not exceed:

  - the largest integer representable by `self.dtype`, i.e.,
    `2**(mantissa_bits+1)` (IEEE 754),
  - the maximum `Tensor` index, i.e., `2**31-1`.
    In other words,
  ```python
  K <= min(2**31-1, {
    tf.float16: 2**11,
    tf.float32: 2**24,
    tf.float64: 2**53 }[param.dtype])

  Note: This condition is validated only when `self.validate_args = True`.
```

Categorical distribution是一种离散分布

又要补数学了

## Categorical distribution

这里参考[这个](https://zhangzhenhu.github.io/blog/glm/source/%E6%A6%82%E7%8E%87%E5%9F%BA%E7%A1%80/content.html#id17)

伯努利分布（Bernoulli distribution）是单个随机变量的一种分布，它只有两个可能的取值。

多努利分布（multinoulli distribution）或者范畴分布（categorical distribution）是单个随机变量的一种分布，它有 k 个可能的取值，不过 k 是有限的。例如人的血型。



```python
dist = Categorical(probs=[0.1, 0.5, 0.4])
n = 1e4
empirical_prob = tf.cast(
tf.histogram_fixed_width(
        dist.sample(int(n)),
        [0., 2],
        nbins=3),
      dtype=tf.float32) / n
  # ==> array([ 0.1005,  0.5037,  0.3958], dtype=float32)
```

一个例子。

这里的prior是权重，也就是创建了一个范畴分布，概率是权重。

# 第三句

```python
mix   = tfp.distributions.MixtureSameFamily(prob, dist)
```

```python
class _MixtureSameFamily(distribution.Distribution):
  """Mixture (same-family) distribution.
  The `MixtureSameFamily` distribution implements a (batch of) mixture
  distribution where all components are from different parameterizations of the
  same distribution type. It is parameterized by a `Categorical` 'selecting
  distribution' (over `k` components) and a components distribution, i.e., a
  `Distribution` with a rightmost batch shape (equal to `[k]`) which indexes
  each (batch of) component.
```

例子

 ```python
  tfd = tfp.distributions
  ### Create a mixture of two scalar Gaussians:
  gm = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(
          probs=[0.3, 0.7]),
      components_distribution=tfd.Normal(
        loc=[-1., 1],       # One for each component.
        scale=[0.1, 0.5]))  # And same here.
 import numpy as np
 x = np.linspace(-2., 3., int(1e4), dtype=np.float32)
 import matplotlib.pyplot as plt
 plt.plot(x, gm.prob(x));
 ```

![image-20220213174110526](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewin202202131741558.png)

这里的这个prob是pdf pdf是概率密度函数，纵坐标是概率密度，而不是概率，cdf对应的纵坐标才是概率，总的积分面积是1

![preview](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewin202202131756011.jpeg)

对密度函数来说，某点，比如x=1的位置，这个x=1邻域的面积，才是这点的概率，如左图所示。当然，邻域宽度无穷小，所以每一点的概率是0，如果一个区域足够宽，它的概率值就显示出来了。

这个面积是1这回事我也差点不信，然后手动算了下

```python
from scipy import integrate
v, err = integrate.quad(gm.prob, -2, 3)
print(v)
```

```
0.9999778336488041
```

确实是1

# 下一句

```python
likelihood = mix.log_prob(y)
```

```python
def log_prob(self, value, name='log_prob', **kwargs):
    """Log probability density/mass function.
    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.
    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    return self._call_log_prob(value, name, **kwargs)
```

我又不懂了 为啥likelihood是logpdf呢，查了查[这个](https://www.zhihu.com/question/54082000)

先看似然函数的定义，他是在给定联合样本值X下关于（未知）参数$\theta$的函数:
$$
L(\theta|x)=f(x|\theta)
$$
这里小x是指联合样本随机变量X取到的值，$\theta$是指未知参数，它属于参数空间。

这里$f(x|\theta)$是一个密度函数，特别的，他表示给定$\theta$下关于样本联合值x的联合密度函数。

所以从定义上，似然函数和密度函数是完全不同的两个**数学对象**：前者是关于![[公式]](https://www.zhihu.com/equation?tex=\theta)的函数，后者是关于![[公式]](https://www.zhihu.com/equation?tex=\textbf{x})的函数。所以这里的等号![[公式]](https://www.zhihu.com/equation?tex=%3D) 理解为**函数值形式**的相等，而不是两个函数本身是同一函数(根据函数相等的定义，函数相等当且仅当定义域相等并且对应关系相等)。

两者的联系则是

（1）如果**x**是离散的随机向量，那么其**概率**密度函数$f(x|\theta)$可改写为$f(x|\theta)=P_{\theta}(X=x)$，即代表了在参数$\theta$下随机向量![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7BX%7D)取到值![[公式]](https://www.zhihu.com/equation?tex=\textbf{x})的**可能性**；并且，**如果**我们发现

![[公式]](https://www.zhihu.com/equation?tex=L(\theta_1+|+\textbf{x}+)+%3D+\mathbb{P}_{\theta_1}(\textbf{X}+%3D+\textbf{x})+>+\mathbb{P}_{\theta_2}(\textbf{X}+%3D+\textbf{x})+%3D+L(\theta_2+|+\textbf{x}))

那么**似然**函数就反应出这样一个**朴素推测**：在参数![[公式]](https://www.zhihu.com/equation?tex=\theta_1)下随机向量![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7BX%7D)取到值![[公式]](https://www.zhihu.com/equation?tex=\textbf{x})的**可能性大于** 在参数![[公式]](https://www.zhihu.com/equation?tex=\theta_2)下随机向量![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7BX%7D)取到值![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7Bx%7D)的**可能性**。换句话说，我们更有理由相信(相对于![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_2)来说)![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_1)

**更有可能**是真实值。这里的可能性由概率来刻画。

（2）如果![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7BX%7D)是连续的随机向量，那么其密度函数![[公式]](https://www.zhihu.com/equation?tex=+f(\textbf{x}+|+\theta))本身（如果在![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7Bx%7D)连续的话）在![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7Bx%7D)处的**概率**为0，为了方便考虑一维情况：给定一个充分小![[公式]](https://www.zhihu.com/equation?tex=\epsilon+>+0)，那么随机变量![[公式]](https://www.zhihu.com/equation?tex=X)取值在![[公式]](https://www.zhihu.com/equation?tex=(x+-+\epsilon%2C+x+%2B+\epsilon))区间内的**概率**即为

![[公式]](https://www.zhihu.com/equation?tex=\mathbb{P}_\theta(x+-+\epsilon+<+X+<+x+%2B+\epsilon)+%3D+\int_{x+-+\epsilon}^{x+%2B+\epsilon}+f(x+|+\theta)+dx+\approx+2+\epsilon+f(x+|+\theta)+%3D+2+\epsilon+L(\theta+|+x))

并且两个未知参数的情况下做比就能约掉![[公式]](https://www.zhihu.com/equation?tex=2\epsilon)，所以和离散情况下的理解一致，只是此时**似然**所表达的那种**可能性**和**概率**![[公式]](https://www.zhihu.com/equation?tex=f(x|\theta)+%3D+0)无关。

综上，**概率**(密度)表达给定![[公式]](https://www.zhihu.com/equation?tex=\theta)下样本随机向量![[公式]](https://www.zhihu.com/equation?tex=\textbf{X}+%3D+\textbf{x})的**可能性**，而**似然**表达了给定样本![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7BX%7D+%3D+%5Ctextbf%7Bx%7D)下参数![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_1)(相对于另外的参数![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_2))为真实值的**可能性**。我们总是对随机变量的取值谈**概率**，而在非贝叶斯统计的角度下，参数是一个实数而非随机变量，所以我们一般不谈一个参数的**概率**。

最后我们再回到![[公式]](https://www.zhihu.com/equation?tex=L(\theta+|+\textbf{x})+%3D+f(\textbf{x}+|+\theta))这个表达。首先我们严格记号，竖线![[公式]](https://www.zhihu.com/equation?tex=|)表示条件概率)或者条件分布，分号![[公式]](https://www.zhihu.com/equation?tex=%3B)表示把参数隔开。所以这个式子的严格书写方式是![[公式]](https://www.zhihu.com/equation?tex=L(\theta+|+\textbf{x})+%3D+f(\textbf{x}+%3B+\theta))因为![[公式]](https://www.zhihu.com/equation?tex=\theta)在右端只当作参数理解。

我没有看懂，说实话，先放这里吧

一些参考

 https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/mixture_same_family.py

https://lulaoshi.info/machine-learning/linear-model/maximum-likelihood-estimation

https://zhangzhenhu.github.io/blog/glm/source/%E6%A6%82%E7%8E%87%E5%9F%BA%E7%A1%80/content.html#id17

# 最后一句

```python
tf.reduce_mean(-likelihood) + tf.add_n([0.] + self.model.losses)
```

tf.reduce_mean是计算likelihood的平均值。

tf.add_n([p1, p2, p3....])函数是实现一个列表的元素的相加。就是输入的对象是一个列表，列表里的元素可以是向量，矩阵，等

self.model.losses 这个我就真的不是特别懂了，除了这里之外没找到有别的地方用过这个



# 跑个debug康康

我好像没有看到这个函数被调用

## MDN代码解析

默认模型参数为

```python
param={'LOO_CV':False, 
 'align':None,'animate':False, 
 'avg_est':False, 
 'batch':128, 
 'benchmark':False, 
 'darktheme':False, 
 'data_loc':'D:/Data/Insitu', 
 'dataset':'all', 'epsilon':0.001,
 
 'filename':None,
 'imputations':5, 
 'l2':0.001, 'lr':0.001, 
 'model_lbl':'', 'model_loc':'Weights', 
 'n_hidden':100, 'n_iter':10000, 
 'n_layers':5, 
 'n_mix':5, 
 'n_redraws':50, 
 'n_rounds':10, 
 'no_bagging':False, 
 'no_load':False, 
 'no_save':False, 
 'plot_loss':False, 'product':'chl', 
 'sat_bands':False, 
 'save_data':False, 
 'save_stats':False, 
 'seed':42, 
 'sensor':'sgli', 
 'silent':False, 
 'sim_loc':'D:/Data/Simulated', 
 'subset':'', 
 'threshold':None, 
 'use_all_ratio':False, 
 'use_auc':False, 
 'use_boosting':False, 
 'use_excl_Rrs':False, 
 'use_kbest':0, 
 'use_noise':False, 
 'use_ratio':False, 
 'use_sim':False, 
 'use_tchlfix':False, 
 'verbose':False}
```

里面有这么几个参数之前没了解过

batch是指从全部样本里取的一批数据。

神经网络这种通常要求大数据量的模型，把所有数据同时扔进模型里训练，计算量太大，内存会爆

一个样本跑一次的话，训练速度又太慢

所以一般找一个合适大小的样本量，可以并行计算加快训练速度，一次跑的数据量又不会过大

imputations是数据插补

'l2':0.001, 'lr':0.001,不解释了

剩下的都差不多

还是有一些写代码的细节，自己需要再开一篇

自己写代码的功底真的好差
