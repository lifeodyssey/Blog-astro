---
title: Robust fit
tags:
  - Research
  - python
categories: 学习笔记
mathjax: true
abbrlink: 7ff46c3b
copyright:
date: 2021-05-13 15:08:38
---

这一篇差不多是作为[这个](https://lifeodyssey.github.io/posts/7decea87.html)的后续

<!-- more -->

# scipy.optimize.least_squares

用来搞Robust fit的有好几个，这里先从scipy的这个函数讲起

先看一下scipy cookbook里面的[这个例子](https://scipy-cookbook.readthedocs.io/items/robust_regression.html)

这个例子里拟合了一个正弦函数

```python
def generate_data(t, A, sigma, omega, noise=0, n_outliers=0, random_state=0):
    y = A * np.exp(-sigma * t) * np.sin(omega * t)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 35
    return y + error
```

确定模型参数

```python
A = 2
sigma = 0.1
omega = 0.1 * 2 * np.pi
x_true = np.array([A, sigma, omega])

noise = 0.1

t_min = 0
t_max = 30
```

将三个离群值放在fitting dataset里

```python
t_train = np.linspace(t_min, t_max, 30)
y_train = generate_data(t_train, A, sigma, omega, noise=noise, n_outliers=4)
```

定义损失函数

```python
def fun(x, t, y):
    return x[0] * np.exp(-x[1] * t) * np.sin(x[2] * t) - y
```

剩下就是一些常规的过程

```python
x0 = np.ones(3)
from scipy.optimize import least_squares
res_lsq = least_squares(fun, x0, args=(t_train, y_train))
res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t_train, y_train))
t_test = np.linspace(t_min, t_max, 300)
y_test = generate_data(t_test, A, sigma, omega)
y_lsq = generate_data(t_test, *res_lsq.x)
y_robust = generate_data(t_test, *res_robust.x)


plt.plot(t_train, y_train, 'o', label='data')
plt.plot(t_test, y_test, label='true')
plt.plot(t_test, y_lsq, label='lsq')
plt.plot(t_test, y_robust, label='robust lsq')
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.legend();
```

![img](data:image/png;base64,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%0A2rREOxtUNVqCoxIr6zV2RESqikstryCiddLOoJBWNKcnDpy+6HT69FGaOCAiUgwKaXIpCmlnUEgr%0Auoqw75uIiJ0ppMmlKKSdQSFNRETKi0KaXIp2HBARERGpIhTSRERERGxIIU1ERETEhhTSRERERGxI%0AIU1ERETEhhTSRERERGxIIU1ERETEhhTSREREqpivvvqKiRMnursMuQSFNBERkSpm7ty5CmkVgEKa%0AiIiInFdOTo67S6jSFNJERESqkLvuuouZM2eyb98+HA4HDoeDxo0bs3jxYhwOB//73/+47777qF27%0ANnXq1AHgzjvvpFGjRue0FRsbS69evQodS05OZtSoUdSvXx9fX19atmzJf//733J5bZWNp7sLEBER%0AkfLz/PPPc/jwYVauXMk333yDaZr4+PiQlpYGwCOPPMLVV1/Nhx9+yIkTJwBr30nDOHfbybOPHTt2%0AjG7dupGdnc2ECROIjIzkxx9/5IEHHiAnJ4eHHnqo7F9gJaKQJiIiUoU0atSIWrVq4e3tTceOHQuO%0AL168GIDOnTvz3nvvlajtSZMmsWfPHtatW0fjxo0B6NWrF6mpqYwfP54HHngAh0ODeEWlkCYiIuJC%0Axvhze5xcxRxrllnbpwwdOrTEj/3xxx/p3LkzERER5OfnFxzv168fU6dOZcOGDbRp08YVZVYJCmki%0AIiIuVB5BqiyFh4eX+LGHDh1i+/bteHl5nXOfYRgcOXKkNKVVOQppIiIiUuB81575+vqed6bnkSNH%0ACA0NLfg5JCSEsLAw3nzzTUzz3LDavHlz1xZbySmkiYiIVDE+Pj5kZWWdc/x8AQ0gIiKCpKQkjhw5%0AQkhICADbt29n8+bNhULagAEDePvtt2nQoEGh41IyunpPRESkimnVqhUpKSm8++67rFy5knXr1gGc%0At/cLYPjw4QDceuutzJ8/n48++oihQ4dSq1atQuc99thj1K5dmyuvvJIpU6awaNEivv32W1577bVS%0AXetWVaknTUREpIq55557WLZsGc8++yxpaWlEREQwffr0C/akNWnShC+++ILnnnuO6667jqioKCZO%0AnMiLL75Y6DGBgYH88ccfTJgwgVdeeYV9+/YRFBRE8+bNGTZsWHm9vErDuFBqtjvDMMyKWruIiFQs%0AhmFcsJdJBC79N3Ly/mJN/dVwp4iIiIgNKaSJiIiI2JBCmoiIiIgNKaSJiIiI2JBCmoiIiIgNKaSJ%0A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%0AiYiIiNiQQpqIiIiIDSmkiYiIiNiQQpqIiIiIDSmkiYiIiNiQbUOaYRgOwzBWGYbxtbtrERERESlv%0Atg1pwKPABncXISIiIuIOtgxphmHUBwYC77u7FhERERF3sGVIAyYCTwKmuwsRERERcQfbhTTDMAYB%0ASaZpJgDGyZuIiIhIleLp7gLOoxswxDCMgYAfUN0wjA9M07z97BPHjRtX8H1sbCyxsbHlVaOIiIjI%0ABS1atIhFixaVqg3DNO07omgYRk/gb6ZpDjnPfaadaxcRERE5xTAMTNMs1uig7YY7RURERMTmPWkX%0Ao540ERERqSjUkyYiIiJSSSikiYiIiNiQQpqIiIiIDSmkiYiIiNiQQpqIiIiIDSmkiYiIiNiQQpqI%0AiIiIDSmkiYiIiNiQQpqIiIiIDSmkiYiIiNiQQpqIiIiIDSmkiYiIiNiQQpqIiIiIDSmkiYiIiNiQ%0AQpqIiIiIDSmkiYiIiNiQQpqIiIiIDSmkiYiIiNiQQpqIiIiIDSmkiYiIiNiQQpqIiIiIDSmkiYiI%0AiNiQQpqIiIiIDSmkiYiIiNiQQpqIiIiIDSmkiYiIiNiQQpqIiIiIDXkW5STDMOYBB4GFwELTNA+W%0AaVUiIiIiVVxRe9JeBzKA/wP2GoaxwTCMtw3DuN4wDP+yK09ERESkajJM0yzeAwwjGOgBjAAGA7nA%0Ag6ZpznZ9eRetwyxu7SIiIiLuYBgGpmkaxXpMaYKOYRiPAr8DbwETTNP8vsSNFf+5FdJERESkQihJ%0ASCvScKdhGOMMw1h18mvjM+5ymqa5EqtnrU9xnlhERERELqyo16R5AE8BkcBawzC2GIaxEuh68v7G%0AwDbXlyciIiJSNRVpdidwADBN07zTMIy/At0Af+A7wzBqAOuAKWVUo4iIiEiVU+Rr0gzD6I4V1H47%0Az31RwAHTNI+5uL6L1aNr0kRERKRCKPeJA+6kkCYiIiIVRZlNHBARERGR8lXUa9KkuNLSYMUKWL8e%0AUlLA6YQ6dSAqCrp2herV3V2hiIiI2Jh60lzJNGHePBg8GBo2hBdegG3bwMMDvL2twPbii1C3LvTt%0AC599Brm57q5aREREbEjXpLnK0qXw8MOQnQ1PPgnXXguBgec/9/hx+O47ePtt2L3bCm4jRoBDmVlE%0ARKQy0sQBd8jNheeeg5kz4fXXix+2fv0VHn/c6mmbOROaNi27WkVERMQtNHGgvKWmwoABsG4drFkD%0A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%0AnaudNw1rbQWwBBhPI9p5rIW01UAfY0x3Y0wi8H1gfpTL1OwZY5JD78IwxqQAFwGbo1uqZsvw5Xki%0A84Efhz7+ETDvq18gZ+xLdR765fmZy1FbD7d84ENr7WPHXFM7j6zj6lztPHKMMe0/Gz42xniBcThz%0AARvczmNqdSd8firBYzgBc5a19sEoF6nZM8b0xOk9szgTIP+qeg8/Y8zLwDeBs4Bi4G5gLvAa0A3Y%0ADUy21pZFq4zNzUnqfCzOvJ0gsAuY9tk8EjkzxphRwDJgE87vEwv8ElgFvIraedidos6vQu08Iowx%0Ag3AWBnhCj9nW2vuMMe1oYDuPuZAmIiIi0hLE2nCniIiISIugkCYiIiLiQgppIiIiIi6kkCYiIiLi%0AQgppIiIiIi6kkCYiIiLiQgppIiIiIi6kkCYiIiLiQgppIiIhxhiPMWarMaZztMsiIqKQJiLyhaFA%0AO2vt/mgXREREIU1E5AtjgfeiXQgREdDZnSIiGGMuA84DrgRWA9uBJ6y1hVEtmIi0aAppIiKAMSYR%0AKAFyrbXbo10eERENd4qIOEYB5QpoIuIWCmkiIo4LgaXRLoSIyGcU0kREHBcCSwCMMaNDw58iIlGj%0AkCYi4hgIfBAKZ9+w1vqiXSARadm0cEBEBDDGPAzUA4eAp6y1VVEukoi0cAppIiIiIi6k4U4RERER%0AF1JIExEREXEhhTQRERERF1JIExEREXEhhTQRERERF1JIExEREXEhhTQRERERF1JIExEREXEhhTQR%0AERERF/p/BgabFaz9QPEAAAAASUVORK5CYII=%0A)

很清楚的可以看出来，这里robust lsq结果明显更接近那个true的线

这里只用了soft l1,在least_squares里还有另外几个

>**loss** str or callable, optional
>
>Determines the loss function. The following keyword values are allowed:
>
>> - ‘linear’ (default) : `rho(z) = z`. Gives a standard least-squares problem.
>> - ‘soft_l1’ : `rho(z) = 2 * ((1 + z)**0.5 - 1)`. The smooth approximation of l1 (absolute value) loss. Usually a good choice for robust least squares.
>> - ‘huber’ : `rho(z) = z if z <= 1 else 2*z**0.5 - 1`. Works similarly to ‘soft_l1’.
>> - ‘cauchy’ : `rho(z) = ln(1 + z)`. Severely weakens outliers influence, but may cause difficulties in optimization process.
>> - ‘arctan’ : `rho(z) = arctan(z)`. Limits a maximum loss on a single residual, has properties similar to ‘cauchy’.
>
>

比较难受的就是这次没法像curve_fit那个函数一样那么好用了

就 我还要自己手动写个cost func

## 另一个例子

这个例子来自于[这儿](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)。

感谢这个例子教会我写cost func

Define the model function as `y = a + b * exp(c * t)`, where t is a predictor variable, y is an observation and a, b, c are parameters to estimate

First, define the function which generates the data with noise and outliers, define the model parameters, and generate data:

```python
>>> def gen_data(t, a, b, c, noise=0, n_outliers=0, random_state=0):
...     y = a + b * np.exp(t * c)
...
...     rnd = np.random.RandomState(random_state)
...     error = noise * rnd.randn(t.size)
...     outliers = rnd.randint(0, t.size, n_outliers)
...     error[outliers] *= 10
...
...     return y + error
...
>>> a = 0.5
>>> b = 2.0
>>> c = -1
>>> t_min = 0
>>> t_max = 10
>>> n_points = 15
...
>>> t_train = np.linspace(t_min, t_max, n_points)
>>> y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)
```

Define function for computing residuals and initial estimate of parameters.

```python
>>> def fun(x, t, y):
...     return x[0] + x[1] * np.exp(x[2] * t) - y
...
>>> x0 = np.array([1.0, 1.0, 0.0])
```

**NOTE**: x0,x1,x2 corresponding to  a,b,c; and t is what we always treat as x

Compute a standard least-squares solution:

```python
>>> res_lsq = least_squares(fun, x0, args=(t_train, y_train))
```

Now compute two solutions with two different robust loss functions. The parameter *f_scale* is set to 0.1, meaning that inlier residuals should not significantly exceed 0.1 (the noise level used).

```python
>>> res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,
...                             args=(t_train, y_train))
>>> res_log = least_squares(fun, x0, loss='cauchy', f_scale=0.1,
...                         args=(t_train, y_train))
```

```python
t_test = np.linspace(t_min, t_max, n_points * 10)
>>> y_true = gen_data(t_test, a, b, c)
>>> y_lsq = gen_data(t_test, *res_lsq.x)
>>> y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
>>> y_log = gen_data(t_test, *res_log.x)
...
>>> import matplotlib.pyplot as plt
>>> plt.plot(t_train, y_train, 'o')
>>> plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
>>> plt.plot(t_test, y_lsq, label='linear loss')
>>> plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
>>> plt.plot(t_test, y_log, label='cauchy loss')
>>> plt.xlabel("t")
>>> plt.ylabel("y")
>>> plt.legend()
>>> plt.show()
```

![scipy-optimize-least_squares-1_00_00](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/scipy-optimize-least_squares-1_00_00.png)

这个例子后面还有一个解决复数优化的问题，可真是牛逼



# scikit learn

这个大体上相同，

```python
from matplotlib import pyplot as plt
import numpy as np

from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)

X = np.random.normal(size=400)
y = np.sin(X)
# Make sure that it X is 2D
X = X[:, np.newaxis]

X_test = np.random.normal(size=200)
y_test = np.sin(X_test)
X_test = X_test[:, np.newaxis]

y_errors = y.copy()
y_errors[::3] = 3

X_errors = X.copy()
X_errors[::3] = 3

y_errors_large = y.copy()
y_errors_large[::3] = 10

X_errors_large = X.copy()
X_errors_large[::3] = 10

estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)),
              ('HuberRegressor', HuberRegressor())]
colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen', 'HuberRegressor': 'black'}
linestyle = {'OLS': '-', 'Theil-Sen': '-.', 'RANSAC': '--', 'HuberRegressor': '--'}
lw = 3

x_plot = np.linspace(X.min(), X.max())
for title, this_X, this_y in [
        ('Modeling Errors Only', X, y),
        ('Corrupt X, Small Deviants', X_errors, y),
        ('Corrupt y, Small Deviants', X, y_errors),
        ('Corrupt X, Large Deviants', X_errors_large, y),
        ('Corrupt y, Large Deviants', X, y_errors_large)]:
    plt.figure(figsize=(5, 4))
    plt.plot(this_X[:, 0], this_y, 'b+')

    for name, estimator in estimators:
        model = make_pipeline(PolynomialFeatures(3), estimator)
        model.fit(this_X, this_y)
        mse = mean_squared_error(model.predict(X_test), y_test)
        y_plot = model.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                 linewidth=lw, label='%s: error = %.3f' % (name, mse))

    legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
    legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                        prop=dict(size='x-small'))
    plt.xlim(-4, 10.2)
    plt.ylim(-2, 10.2)
    plt.title(title)
plt.show()
```

有个PolynomialFeatures(3)的意思是，可以从0阶多项式拟合到3阶多项式

>Robust fitting is demoed in different situations:
>
>- No measurement errors, only modelling errors (fitting a sine with a polynomial)
>- Measurement errors in X
>- Measurement errors in y
>
>The median absolute deviation to non corrupt new data is used to judge the quality of the prediction.
>
>What we can see that:
>
>- RANSAC is good for strong outliers in the y direction
>- TheilSen is good for small outliers, both in direction X and y, but has a break point above which it performs worse than OLS.
>- The scores of HuberRegressor may not be compared directly to both TheilSen and RANSAC because it does not attempt to completely filter the outliers but lessen their effect.

# iterative bi-square method

其实我最想找的是这个，这个来源于[这篇文献]( doi:10.3390/rs12010077)

但是找来找去都没找到python的包

啊 又逼着我改用R了