---
title: Probabilistic Method in Machine Learning
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
  - tensorflow
categories:
  - Learning Notes
abbrlink: 62469dc9
slug: probabilistic-method-in-machine-learning
date: 2022-02-12 14:24:22
mathjax: true
copyright: true
lang: en
---

Statistical methods in machine learning, mainly following from the Mixture Density Network article.

<!-- more -->

In the MDN article, the main functions used are tfp.distributions.MixtureSameFamily(prob, dist),

mix.log_prob(y)

MixtureSameFamily, a categorical,

```python
def loss(self, y, output):
    prior, mu, scale = self._parse_outputs(output)
    # Get three parameters from output
    dist  = getattr(tfp.distributions, self.distribution)(mu, scale)
    prob  = tfp.distributions.Categorical(probs=prior)
    mix   = tfp.distributions.MixtureSameFamily(prob, dist)
    likelihood = mix.log_prob(y)
    return tf.reduce_mean(-likelihood) + tf.add_n([0.] + self.model.losses)
```

This is the function.

Let's look at what's being called inside in detail.

# First Line

```python
 dist  = getattr(tfp.distributions, self.distribution)(mu, scale)
```

getattr gets the attribute value of an object

```python
getattr(object, name[, default])
```

- object -- The object.
- name -- String, object attribute.
- default -- Default return value. If not provided, AttributeError will be triggered when there's no corresponding attribute.

But here it's using getattr to call a function.

self.distribution equals 'MultivariateNormalTriL'. This line actually calls a 'MultivariateNormalTriL' function instance from tfp.distributions, then passes (mu, scale) as parameters to this function.

This function is:

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

Simply put, it takes variance and mean as two parameters to calculate a multivariate normal distribution.

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
```

```python
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

# Second Line

After getting this distribution:

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

Categorical distribution is a discrete distribution.

Time to review some math again.

## Categorical Distribution

Reference [this](https://zhangzhenhu.github.io/blog/glm/source/%E6%A6%82%E7%8E%87%E5%9F%BA%E7%A1%80/content.html#id17)

The Bernoulli distribution is a distribution of a single random variable with only two possible values.

The multinoulli distribution or categorical distribution is a distribution of a single random variable with k possible values, where k is finite. For example, human blood types.

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

An example.

Here, prior is the weight, meaning a categorical distribution is created with probabilities as weights.

# Third Line

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

Example:

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

Here, prob is the pdf. PDF is the probability density function, where the y-axis is probability density, not probability. The CDF's y-axis corresponds to probability, and the total integral area is 1.

![preview](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewin202202131756011.jpeg)

For a density function, at a certain point like x=1, the area of the neighborhood around x=1 is the probability at that point, as shown in the left figure. Of course, when the neighborhood width is infinitely small, the probability at each point is 0. If a region is wide enough, its probability value becomes apparent.

I almost didn't believe the area is 1, so I calculated it manually:

```python
from scipy import integrate
v, err = integrate.quad(gm.prob, -2, 3)
print(v)
```

```
0.9999778336488041
```

Indeed it's 1.

# Next Line

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

I don't understand again - why is likelihood the log pdf? I looked up [this](https://www.zhihu.com/question/54082000).

First, look at the definition of the likelihood function. It's a function of the (unknown) parameter $\theta$ given the joint sample value X:
$$
L(\theta|x)=f(x|\theta)
$$
Here, lowercase x refers to the value taken by the joint sample random variable X, and $\theta$ refers to the unknown parameter belonging to the parameter space.

Here $f(x|\theta)$ is a density function. Specifically, it represents the joint density function of the joint sample value x given $\theta$.

So by definition, the likelihood function and density function are completely different **mathematical objects**: the former is a function of $\theta$, the latter is a function of $\textbf{x}$. So the equals sign here should be understood as equality of **function values**, not that the two functions themselves are the same function.

The connection between them is:

(1) If **x** is a discrete random vector, then its **probability** density function $f(x|\theta)$ can be rewritten as $f(x|\theta)=P_{\theta}(X=x)$, representing the **possibility** of random vector X taking value x under parameter $\theta$.

(2) If X is a continuous random vector, then its density function $f(x|\theta)$ itself has **probability** 0 at x (if continuous at x).

In summary, **probability** (density) expresses the **possibility** of sample random vector X = x given $\theta$, while **likelihood** expresses the **possibility** that parameter $\theta_1$ (relative to another parameter $\theta_2$) is the true value given sample X = x.

I didn't fully understand this, to be honest. Let's leave it here for now.

Some references:

https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/mixture_same_family.py

https://lulaoshi.info/machine-learning/linear-model/maximum-likelihood-estimation

https://zhangzhenhu.github.io/blog/glm/source/%E6%A6%82%E7%8E%87%E5%9F%BA%E7%A1%80/content.html#id17

# Last Line

```python
tf.reduce_mean(-likelihood) + tf.add_n([0.] + self.model.losses)
```

tf.reduce_mean calculates the mean of likelihood.

tf.add_n([p1, p2, p3....]) implements addition of elements in a list. The input object is a list, and the elements can be vectors, matrices, etc.

self.model.losses - I really don't understand this one. I couldn't find it used anywhere else besides here.

# Running Debug to See

I don't seem to see this function being called.

## MDN Code Analysis

Default model parameters are:

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

There are a few parameters I wasn't familiar with before:

batch refers to a batch of data taken from all samples.

Neural networks typically require large amounts of data. Throwing all data into the model at once would be too computationally expensive and would blow up memory.

Running one sample at a time would make training too slow.

So we generally find an appropriate sample size that allows parallel computation to speed up training without processing too much data at once.

imputations is data imputation.

'l2':0.001, 'lr':0.001 - no explanation needed.

The rest are similar.

There are still some coding details that I need to write about in another article.

My coding skills are really lacking.

