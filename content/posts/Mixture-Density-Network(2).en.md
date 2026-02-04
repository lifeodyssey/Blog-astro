---
title: Mixture Density Network (2)
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
categories:
  - Learning Notes
mathjax: true
abbrlink: 8bb8e4ec
slug: mixture-density-network-2
copyright: true
date: 2022-02-10 13:23:00
lang: en
---
It's too long, so I'm splitting it into parts.
<!-- more -->

# Math Review

## What is Maximum Likelihood

The likelihood is the possibility that fits a distribution.

For example, if you flip a fair coin 10 times, what's the probability of getting 6 heads? This probability is the "probability" - the possibility of a certain event occurring.

Likelihood, on the other hand, is given a certain result, finding the possibility that it comes from a certain distribution.

For example, if you flip a coin 10 times and get 6 heads, what's the likelihood that the coin is fair?

Another example: suppose after statistics, the final exam scores for functional analysis follow $N(80,4^2)$. What's the probability that Xiao Huang scored 90? This calculates probability.

Another example: Xiao Huang scored 90, what's the probability that the exam follows $N(80,4^2)$? This is likelihood.

We use this to select parameters that maximize the likelihood.

So, how do we calculate likelihood?

## How to Get Likelihood

This is actually maximum likelihood estimation.

I just realized I learned this concept in my sophomore year, but I've completely forgotten it now.

Let's review it.

Let's use a simpler problem to understand this concept. The simplest distribution is the binomial distribution. We'll use coin flipping as an example.

When I use coin flipping as an example, there's an implicit requirement: each event is independent. To use the concept of likelihood, each event must be independent.

For a fair coin, the probability of heads is 0.5 - this is probability.

For a coin that was flipped once and landed heads, what's the probability that the next flip will be heads (i.e., determining the parameter of this binomial distribution)? This is likelihood.

But obviously, we can't calculate the likelihood from this example because the sample is too small. Intuitively, we need enough samples to replace probability with frequency. More academically, this is the Law of Large Numbers and Central Limit Theorem.

### Law of Large Numbers and Central Limit Theorem

This is the most fundamental concept in mathematical statistics. Since we're reviewing, let's review thoroughly.

**Example:** Flip a fair coin where heads and tails have equal probability. Let the frequency of heads be $v_n=S_n/n$, where $S_n$ is the number of heads and n is the total number of flips. As we keep flipping, we'll observe two phenomena in the frequency sequence ${v_n}$:

1. The absolute deviation $|v_n-p|$ of frequency ${v_n}$ from probability p tends to decrease as n increases, but we can't say it converges to 0.
2. Due to the randomness of frequency, the absolute deviation $|v_n-p|$ varies. While we can't rule out large deviations, as n increases, the possibility of large deviations becomes smaller. **This is a new concept of limit.**

Definition:

If for any $\varepsilon$, we have
$$
\lim_{n\to\infty}P(|\xi_n-\xi|\ge\varepsilon)=0
$$
then we say the random variable sequence {$\xi_n,n\in N$} converges in probability to random variable $\xi$, denoted as $\xi_n\to^{P}\xi$

Convergence in probability means: the possibility that the absolute deviation of $\xi_n$ from $\xi$ is not less than any given amount will become smaller as n increases. Conversely, the possibility that $|\xi_n-\xi|$ is less than any given amount approaches 1 as n increases.

Let's rewrite it:
$$
\lim_{n\to\infty}P(|\xi_n-\xi|\leq\varepsilon)=1
$$
For example, the sequence $v_n$ will increasingly tend toward a certain number.

This is actually called the Law of Large Numbers.

Theorem:

In n independent repeated experiments, event A occurs $k_n$ times, and $P(A)=p$. For any $\varepsilon>0$:
$$
\lim_{n\to\infty}P(|\frac{k_n}{n}-p|<\varepsilon)=0
$$
This is Bernoulli's Law of Large Numbers, which can be proven using Chebyshev's inequality.

Bernoulli's Law of Large Numbers is a special case for binomial distributions.

There are also:

| Chebyshev's Law | Independent $X_{1}, X_{2}, \cdots$ with bounded variance | Sample mean converges to expected mean |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Bernoulli's Law | $n_{A} \sim B(n, p)$ | $\frac{n_{A}}{n}\stackrel{P}{\longrightarrow}p$ |
| Khinchin's Law | Independent identically distributed $X_{1}, X_{2}, \cdots$ with expectation $\mu$ | Sample mean converges to $\mu$ |

Chebyshev's Law is the most general - it doesn't require identical distributions. Khinchin's Law is a special case when distributions are identical. Bernoulli's Law is when the distribution is binomial. The essence of the Law of Large Numbers: sample mean equals population mean.

In probability theory, the **Central Limit Theorem** describes the average of a sequence of independent identically distributed random variables.

Specifically, the Law of Large Numbers states that for independent identically distributed random variables, as n→∞, the mean almost surely converges to the expectation. The Central Limit Theorem states that the difference between the mean and expectation approximately follows a normal distribution N(0,σ²) scaled by 1/n, where σ² is the variance of the original random variables.

**Central Limit Theorem for Identical Distributions:** Let X1, X2, …, Xn be mutually independent with the same distribution, having mathematical expectation and variance. Then for any x, we have the standardized sum converging to the standard normal distribution.

## So How Do We Actually Calculate It?

Let's try flipping a coin 10 times. The likelihood function is usually denoted as *L* (for Likelihood). Observing "6 heads, 4 tails", the likelihood function for different values of parameter θ is:
$$
L(\theta;6H4T)=C^6_{10}*\theta^6*(1-\theta)^4
$$
The graph of this formula is shown below. From the graph: when parameter θ is 0.6, the likelihood function is maximized. For other parameter values, the probability of "6 heads, 4 tails" is relatively smaller. In this bet, I would guess the next flip will be heads, because based on observations, the coin likely has a 0.6 probability of heads.

!["6H4T" likelihood function](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewin202202131440250.png)

Generalizing to more general scenarios, the general form of the likelihood function can be expressed as the product of probabilities of each sample occurring.

![image-20220213144122733](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewin202202131441773.png)

This also involves maximum likelihood estimation, but this paper's loss only uses maximum likelihood, so I'll skip that for now.

References:
lulaoshi.info/machine-learning/linear-model/maximum-likelihood-estimation
https://www.cnblogs.com/BlairGrowing/p/14877125.html

# Looking at the Code

Now let's look at the paper's source code.

It's so long... and written in TensorFlow... I'll have to rewrite it myself...

The MDN class handles multi-output, full (symmetric) covariance with parameters including:
- n_mix: Number of mixtures (default=5)
- hidden: Layers and hidden units (default=[100]*5)
- lr: Learning rate (default=1e-3)
- l2: L2 regularization (default=1e-3)
- n_iter: Training iterations (default=1e4)
- batch: Minibatch size (default=128)

The key methods in the MDN class:

1. **`__init__`**: Constructor that initializes all parameters
2. **`_predict_chunk`**: Generates estimates for a subset of data
3. **`predict`**: Main prediction interface that wraps _predict_chunk
4. **`extract_predictions`**: Extracts model predictions from coefficients
5. **`fit`**: Training method that handles data preprocessing and model training
6. **`build`**: Constructs the neural network architecture
7. **`loss`**: Calculates the loss using likelihood
