---
title: Mixture Density Network(2)
tags:
  - 机器学习
  - Inversion
  - Deep Learning
categories:
  - 学习笔记
mathjax: true
abbrlink: 8bb8e4ec
copyright: true
date: 2022-02-10 13:23:00
---
太长了，分P
<!-- more -->

# 数学补课

## What is maximum likelihood

the likelihood is the possibility that fit a distribution

例えば 抛一枚匀质硬币，抛10次，6次正面向上的可能性多大？ 这里的可能性是probability 是某个时间发生的可能性

而似然值是给定某一结果，求是这个分布的可能性。

例如 抛一枚硬币，抛10次，结果是6次正面向上，其是匀质的可能性多大？

再举个例子，比如经过统计，这次期末考试泛函分析的成绩服从$N(80,4^2)$，问小黄这次考了90分的概率是多少。这个计算的就是probability

再比如，小黄这次考了90，问这次期末考试泛函服从$N(80,4^2 )$的概率是多少，这个就是likelihood。

我们要用的就是这个，来选取一个让likelihood最大的参数。

那么，怎么计算likelihood呢？

## How to get likelihood

其实这个问题就是极大似然估计。

我到这里才发现这个概念我大二就学过，然而现在完全忘记了。

就当复习了。

我们来换个更简单一点的问题来理解这个概念，最简单的分布是二项分布。这里我们就拿抛硬币为例。

在我拿硬币举例子的时候其实就暗含了一个要求，就是每个事件是独立的。只要能够用到likelihood这个概念，就得要求每个事件都是独立的。

一个质量均匀的硬币，抛出来是正面的概率 是0.5这里这个概率是probability。

一个硬币，抛了一次是正面，求下一次抛出来是正面的概率（即确定这个二项分布的参数），这个是likelihood。

但是我们举得这个例子显然无法计算出来likelihood，因为样本太少了。从直觉来想，我们需要做足够多的采样，用频率来替代概率。更学术一点，这个东西其实就是大数定律和中心极限定理。

###　大数定律和中心极限定理

这是整个数理统计最基础的东西了，既然复习我们就复习到底

**例子：** 抛一个均匀的硬币，正面朝上和反面朝上的概率是相等的。我们把正面朝上的频率记为 $v_n=S_n/n$，其中$S_n$为正面朝上出现的次数，n为抛硬币的总次数，那么当把硬币一直抛下去，我们会发现频率序列 ${v_n}$ 会出现两个现象：

1. 频率 ${v_n}$ 对频率p的绝对偏差 $|v_n-p|$ 将随着n的增大而呈现逐渐减小的趋势，但无法说它就收敛域0
2. 由于频率的随机性，绝对偏差  $|v_n-p|$ 时大时小，虽然我们无法排除大偏差发生的可能性，但随着n的不断增大，大偏差发生的可能性会越来越小。 **这是一种新的极限概念**

定义

如果对于任何$\varepsilon$，都有
$$
\lim_{n\to\infty}P(|\xi_n-\xi|\ge\varepsilon)=0
$$
那么我们就称随机变量序列{$\xi_n,n\in N$}依概率收敛到随机变量$\xi$，记为$\xi_n\to^{P}\xi$

依概率收敛的含义是：$ \xi_n $ 对$ \xi $ 的绝对偏差不小于任意给定量的可能性将随着n的增大而越来越小。反过来说，绝对偏差$|\xi_n-\xi|$小于任一给定量的可能性将随着 n的增大而越来越接近与1。

我们来换个写法
$$
\lim_{n\to\infty}P(|\xi_n-\xi|\leq\varepsilon)=1
$$
举个例子就是$v_n$这个序列将越来越倾向于某一个数。

其实这个东西，就是叫大数定理

定理

在n次独立重复实验中，事件A发生了$k_n$次，且$P(A)=p$，则对任意$\varepsilon>0$，有
$$
\lim_{n\to\infty}P(|\frac{k_n}{n}-p|<\varepsilon)=0
$$
这个东西就是伯努利大数定律，可以利用切比雪夫不等式来证明。

很容易可以看到，伯努利大数定律是针对二项分布的一个特殊情况。

除了这个还有。

| 切比雪夫大数定律 | ![X_{1}, X_{2}, \cdots](https://math.fivecakes.com/?latex=X_%7B1%7D%2C%20X_%7B2%7D%2C%20%5Ccdots) 独立，存在期望及方差，且方差有界 | ![\frac{1}{n} \sum_{k=1}^{n} X_{k}\stackrel{P}{\longrightarrow}\frac{1}{n} \sum_{k=1}^{n} E\left(X_{k}\right)](https://math.fivecakes.com/?latex=%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bk%3D1%7D%5E%7Bn%7D%20X_%7Bk%7D%5Cstackrel%7BP%7D%7B%5Clongrightarrow%7D%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bk%3D1%7D%5E%7Bn%7D%20E%5Cleft(X_%7Bk%7D%5Cright)) |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 伯努利大数定律   | ![n_{A} \sim B(n, p)](https://math.fivecakes.com/?latex=n_%7BA%7D%20%5Csim%20B(n%2C%20p)) | ![\frac{n_{A}}{n}\stackrel{P}{\longrightarrow}p](https://math.fivecakes.com/?latex=%5Cfrac%7Bn_%7BA%7D%7D%7Bn%7D%5Cstackrel%7BP%7D%7B%5Clongrightarrow%7Dp) |
| 辛钦大数定律     | ![X_{1}, X_{2}, \cdots](https://math.fivecakes.com/?latex=X_%7B1%7D%2C%20X_%7B2%7D%2C%20%5Ccdots) 独立同分布，期望为![\mu](https://math.fivecakes.com/?latex=%5Cmu) | ![\frac{1}{n} \sum_{k=1}^{n} X_{k}\stackrel{P}{\longrightarrow}\mu](https://math.fivecakes.com/?latex=\frac{1}{n} \sum_{k%3D1}^{n} X_{k}\stackrel{P}{\longrightarrow}\mu) |

切比雪夫大数定律是最宽泛的大数定律，不需要每个序列具有相同的分布；辛钦大数定律则是在他们分布相同的情况下的特殊情况，没有规定是什么分布；伯努利大数定律则是这个分布是二项分布的特殊情况。大数定律的左边的本质是求整个随机变量分布列的平均数，右边则是总体的平均值（期望），即样本的平均值等于总体的平均值。

在概率论中, **中心极限定理**是对一列独立同分布的随机变量之平均值的描述.

具体而言, 大数定律表明, 对一列独立同分布的随机变量而言, 当随机变量的个数 *n*→∞ 时, 其均值几乎必然收敛于其期望. 中心极限定理则表明, 其均值与期望的差大约满足 1/*n* 倍的正态分布 N(0,*σ*2), 其中 *σ*2 是原来随机变量的方差.

**同分布的中心极限定理** 设 X1, X2, …, Xn 相互独立，服从同一分布，具有数学期望和方差：

[![img](https://db.yihui.org/hexun/b_4FE2288307FFF259.jpg)](https://db.yihui.org/hexun/b_4FE2288307FFF259.jpg)

则对任意的 x，恒有

[![img](https://db.yihui.org/hexun/b_94BC3E85BB876C97.jpg)](https://db.yihui.org/hexun/b_94BC3E85BB876C97.jpg)



## 说了那么多 到底怎么算呢

我们这里来抛十次硬币试试，似然函数通常用*L*表示，对应英文Likelihood。观察到抛硬币“6正4反”的事实，硬币参数*θ*取不同值时，似然函数表示为：
$$
L(\theta;6正4反)=C^6_{10}*\theta^6*(1-\theta)^4
$$
这个公式的图形如下图所示。从图中可以看出：参数θ*为0.6时，似然函数最大，参数为其他值时，“6正4反”发生的概率都相对更小。在这个赌局中，我会猜测下次硬币为正，因为根据已有观察，硬币很可能以0.6的概率为正。

![“6正4反”的似然函数](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewin202202131440250.png)

推广到更为一般的场景，似然函数的一般形式可以用下面公式来表示，也就是之前提到的，各个样本发生的概率的乘积。

![image-20220213144122733](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/gamewin202202131441773.png)

这里还涉及到一个东西叫极大似然估计，不过好像这篇论文的loss就只是用了极大似然值，先在这里省略一下

参考资料

lulaoshi.info/machine-learning/linear-model/maximum-likelihood-estimation

https://www.cnblogs.com/BlairGrowing/p/14877125.html

# 来看代码

然后再来看看论文的源代码.

写的好长...而且是用tensorflow写的..自己这不是得重写了...

```python
class MDN:
	''' Mixture Density Network which handles multi-output, full (symmetric) covariance.
	Parameters
	----------
	n_mix : int, optional (default=5)
		Number of mixtures used in the gaussian mixture model.
	hidden : list, optional (default=[100, 100, 100, 100, 100])
		Number of layers and hidden units per layer in the neural network.
	lr : float, optional (default=1e-3)
		Learning rate for the model.
	l2 : float, optional (default=1e-3)
		L2 regularization scale for the model weights.
	n_iter : int, optional (default=1e4)
		Number of iterations to train the model for 
	batch : int, optional (default=128)
		Size of the minibatches for stochastic optimization.
	imputations : int, optional (default=5)
		Number of samples used in multiple imputation when handling NaN
		target values during training. More samples results in a higher
		accuracy for the likelihood estimate, but takes longer and may
		result in overfitting. Assumption is that any missing data is 
		MAR / MCAR, in order to allow a multiple imputation approach.
	epsilon : float, optional (default=1e-3)
		Normalization constant added to diagonal of the covariance matrix.
	activation : str, optional (default=relu)
		Activation function applied to hidden layers.
	scalerx : transformer, optional (default=IdentityTransformer)
		Transformer which has fit, transform, and inverse_transform methods
		(i.e. follows the format of sklearn transformers). Scales the x 
		values prior to training / prediction. Stored along with the saved
		model in order to have consistent inputs to the model.
	scalery : transformer, optional (default=IdentityTransformer)
		Transformer which has fit, transform, and inverse_transform methods
		(i.e. follows the format of sklearn transformers). Scales the y 
		values prior to training, and the output values after prediction. 
		Stored along with the saved model in order to have consistent 
		outputs from the model.
	model_path : pathlib.Path, optional (default=./Weights)
		Folder location to store saved models.
	model_name : str, optional (default=MDN)
		Name to assign to the model. 
	no_load : bool, optional (default=False)
		If true, train a new model rather than loading a previously 
		trained one.
	no_save : bool, optional (default=False)
		If true, do not save the model when training is completed.
	seed : int, optional (default=None)
		Random seed. If set, ensure consistent output.
	verbose : bool, optional (default=False)
		If true, print various information while loading / training.
	debug : bool, optional (default=False)
		If true, use control flow dependencies to determine where NaN
		values are entering the model. Model runs slower with this 
		parameter set to true.
	'''
	distribution = 'MultivariateNormalTriL'

	def __init__(self, n_mix=5, hidden=[100]*5, lr=1e-3, l2=1e-3, n_iter=1e4,
				 batch=128, imputations=5, epsilon=1e-3,
				 activation='relu',
				 scalerx=None, scalery=None, 
				 model_path='Weights', model_name='MDN',
				 no_load=False, no_save=False,
				 seed=None, verbose=False, debug=False, **kwargs):

		config = initialize_random_states(seed)
		config.update({
			'n_mix'        : n_mix,
			'hidden'       : list(np.atleast_1d(hidden)),
			'lr'           : lr,
			'l2'           : l2,
			'n_iter'       : n_iter,
			'batch'        : batch,
			'imputations'  : imputations,
			'epsilon'      : epsilon,
			'activation'   : activation,
			'scalerx'      : scalerx if scalerx is not None else IdentityTransformer(),
			'scalery'      : scalery if scalery is not None else IdentityTransformer(),
			'model_path'   : Path(model_path),
			'model_name'   : model_name,
			'no_load'      : no_load,
			'no_save'      : no_save,
			'seed'         : seed,
			'verbose'      : verbose,
			'debug'        : debug,
		})
		self.set_config(config)
#前面 构造函数 定义变量
		for k in kwargs: 
			warnings.warn(f'Unused keyword given to MDN: "{k}"', UserWarning)


	def _predict_chunk(self, X, return_coefs=False, use_gpu=False, **kwargs):
		''' Generates estimates for the given set. X may be only a subset of the full
			data, which speeds up the prediction process and limits memory consumption.
		
			use_gpu : bool, optional (default=False)
				Use the GPU to generate estimates if True, otherwise use the CPU.
			 '''
		with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
#确定位置
			model_out = self.model( self.scalerx.transform(ensure_format(X)) )
    #先处理成一样的类型
			coefs_out = self.get_coefs(model_out)#解析模型输出
			outputs   = self.extract_predictions(coefs_out, **kwargs)
	# 这个是得到最后的输出，
			if return_coefs: 
				return outputs, [c.numpy() for c in coefs_out]
			return outputs


	@ignore_warnings
	def predict(self, X, chunk_size=1e5, return_coefs=False, **kwargs):
        #这个是预测的主函数
		'''
		Top level interface to get predictions for a given dataset, which wraps _predict_chunk 
		to generate estimates in smaller chunks. See the docstring of extract_predictions() for 
		a description of other keyword parameters that can be given. 
	
		chunk_size : int, optional (default=1e5)
			Controls the size of chunks which are estimated by the model. If None is passed,
			chunking is not used and the model is given all of the X dataset at once. 
		return_coefs : bool, optional (default=False)
			If True, return the estimated coefficients (prior, mu, sigma) along with the 
			other requested outputs. Note that rescaling the coefficients using scalerx/y
			is left up to the user, as calculations involving sigma must be performed in 
			the basis learned by the model.
		'''
		chunk_size    = int(chunk_size or len(X))
		partial_coefs = []
		partial_estim = []

		for i in trange(0, len(X), chunk_size, disable=not self.verbose):
			chunk_est, chunk_coef = self._predict_chunk(X[i:i+chunk_size], return_coefs=True, **kwargs)
            # 把总的数据分成好几块来进行预测
            #每个小块都是在predict_chunk里预测的
			partial_coefs.append(chunk_coef)
			partial_estim.append( np.array(chunk_est, ndmin=3) )

		coefs = [np.vstack(c) for c in zip(*partial_coefs)]
		preds = np.hstack(partial_estim)

		if return_coefs:
			return preds, coefs 
		return preds


	def extract_predictions(self, coefs, confidence_interval=None, threshold=None, avg_est=False):
		'''
		Function used to extract model predictions from the given set of 
		coefficients. Users should call the predict() method instead, if
		predictions from input data are needed. 
		confidence_interval : float, optional (default=None)
			If a confidence interval value is given, then this function
			returns (along with the predictions) the upper and lower 
			{confidence_interval*100}% confidence bounds around the prediction.
		
		threshold : float, optional (default=None)
			If set, the model outputs the maximum prior estimate when the prior
			probability is above this threshold; and outputs the average estimate
			when below the threshold. Any passed value should be in the range (0, 1],
			though the sign of the threshold can be negative in order to switch the
			estimates (i.e. negative threshold would output average estimate when prior
			is greater than the (absolute) value).  
		avg_est : bool, optional (default=False)
			If true, model outputs the prior probability weighted mean as the
			estimate. Otherwise, model outputs the maximum prior estimate.
		'''
		assert(confidence_interval is None or (0 < confidence_interval < 1)), 'confidence_interval must be in the range (0,1)'
		assert(threshold is None or (0 < threshold <= 1)), 'threshold must be in the range (0,1]'
#检查是不是在0-1
		target = ('avg' if avg_est else 'top') if threshold is None else 'threshold'# 选一个方式来处理模型的输出，比如avg
		output = getattr(self, f'_get_{target}_estimate')(coefs)
        #然后处理输出
		scale  = lambda x: self.scalery.inverse_transform(x.numpy())
		
		if confidence_interval is not None:
			assert(threshold is None), f'Cannot calculate confidence on thresholded estimates'
			confidence = getattr(self, f'_get_{target}_confidence')(coefs, confidence_interval)
			upper_bar  = output + confidence
			lower_bar  = output - confidence
			return scale(output), scale(upper_bar), scale(lower_bar)
        #k
		return scale(output)


	@ignore_warnings
	def fit(self, X, Y, output_slices=None, **kwargs):
		with get_device(self.config):#数据放在CPU还是GPU 
			checkpoint = self.model_path.joinpath('checkpoint')
#加载模型文件
			if checkpoint.exists() and not self.no_load:
				if self.verbose: print(f'Restoring model weights from {checkpoint}')
				self.load()
#如果模型存在的话，就加载，不存在的话就不加载
			elif self.no_load and X is None:
				raise Exception('Model exists, but no_load is set and no training data was given.')

			elif X is not None and Y is not None:	
				self.scalerx.fit( ensure_format(X), ensure_format(Y) )
				self.scalery.fit( ensure_format(Y) )
# 对数据做一下预处理，看是不是符合要求，转换函数在前面有定义，在product_estimation里面自定义的
				# Gather all data (train, validation, test, ...) into singular object
				datasets = kwargs['datasets'] = kwargs.get('datasets', {})
				datasets.update({'train': {'x' : X, 'y': Y}})
# 分训练集和测试机
				for key, data in datasets.items(): 
					if data['x'] is not None:
						datasets[key].update({
							'x_t' : self.scalerx.transform( ensure_format(data['x']) ),
							'y_t' : self.scalery.transform( ensure_format(data['y']) ),
						})
#再次转换
assert(np.isfinite(datasets['train']['x_t']).all()), 'NaN values found in X training data'
#数据检查
				self.update_config({
					'output_slices' : output_slices or {'': slice(None)},
					'n_inputs'      : datasets['train']['x_t'].shape[1],
					'n_targets'     : datasets['train']['y_t'].shape[1],
				})
				self.build()

				callbacks = []
				model_kws = {
					'batch_size' : self.batch, 
					'epochs'     : max(1, int(self.n_iter / max(1, len(X) / self.batch))),
					'verbose'    : 0, 
					'callbacks'  : callbacks,
				}

				if self.verbose:
					callbacks.append( TqdmCallback(model_kws['epochs'], data_size=len(X), batch_size=self.batch) )

				if self.debug:
					callbacks.append( tf.keras.callbacks.TensorBoard(histogram_freq=1, profile_batch=(2,60)) )

				if 'args' in kwargs:

					if getattr(kwargs['args'], 'plot_loss', False):
						callbacks.append( PlottingCallback(kwargs['args'], datasets, self) )

					if getattr(kwargs['args'], 'save_stats', False):
						callbacks.append( StatsCallback(kwargs['args'], datasets, self) )

					if getattr(kwargs['args'], 'best_epoch', False):
						if 'valid' in datasets and 'x_t' in datasets['valid']:
							model_kws['validation_data'] = (datasets['valid']['x_t'], datasets['valid']['y_t'])
							callbacks.append( ModelCheckpoint(self.model_path) )

				self.model.fit(datasets['train']['x_t'], datasets['train']['y_t'], **model_kws)

				if not self.no_save:
					self.save()

			else:
				raise Exception(f"No trained model exists at: \n{self.model_path}")
			return self 

# 网络构建的部分在这里
#全连接 输入层 隐藏层 激活函数 输出层

	def build(self):
		layer_kwargs = {
			'activation'         : self.activation,#激活函数
			'kernel_regularizer' : tf.keras.regularizers.l2(self.l2),#一个正则项
			'bias_regularizer'   : tf.keras.regularizers.l2(self.l2),#另一个正则项
			# 'kernel_initializer' : tf.keras.initializers.LecunNormal(),
			# 'bias_initializer'   : tf.keras.initializers.LecunNormal(),
		}
		mixture_kwargs = {
			'n_mix'     : self.n_mix,#整个模型输出几个高斯参数
			'n_targets' : self.n_targets,#在前面都定义过
			'epsilon'   : self.epsilon,
		}
		mixture_kwargs.update(layer_kwargs)
		#字典添加到上面
		create_layer = lambda inp, out: tf.keras.layers.Dense(out, input_shape=(inp,), **layer_kwargs)# 第一层的全连接的方式，
		model_layers = [create_layer(inp, out) for inp, out in zip([self.n_inputs] + self.hidden[:-1], self.hidden)]#把第一层全连接组合起来，
		output_layer = MixtureLayer(**mixture_kwargs)#搭建混合密度网络
		#做初始化
		# Define yscaler.inverse_transform as a tensorflow function, and estimate extraction from outputs
		# yscaler_a   = self.scalery.scalers[-1].min_
		# yscaler_b   = self.scalery.scalers[-1].scale_
		# inv_scaler  = lambda y: tf.math.exp((tf.reshape(y, shape=[-1]) - yscaler_a) / yscaler_b) 
		# extract_est = lambda z: self._get_top_estimate( self._parse_outputs(z) )
		
		optimizer  = tf.keras.optimizers.Adam(self.lr)#优化器
		self.model = tf.keras.Sequential(model_layers + [output_layer], name=self.model_name)#网络组合起来
		self.model.compile(loss=self.loss, optimizer=optimizer, metrics=[])#[MSA(extract_est, inv_scaler)])#和前面的模型叠加进行训练
		

	@tf.function
    
	def loss(self, y, output):
		prior, mu, scale = self._parse_outputs(output) 
        #对输出做解析
		dist  = getattr(tfp.distributions, self.distribution)(mu, scale)
        #选出来正态分布
		prob  = tfp.distributions.Categorical(probs=prior)
        #类别分布
		mix   = tfp.distributions.MixtureSameFamily(prob, dist)
		#把五个分布结合起来变成一个新的分布
		def impute(mix, y, N):
			# summation  = tf.zeros(tf.shape(y)[0])
			# imputation = lambda i, s: [i+1, tf.add(s, mix.log_prob(tf.where(tf.math.is_nan(y), mix.sample(), y)))]
			# return tf.while_loop(lambda i, x: i < N, imputation, (0, summation), maximum_iterations=N, parallel_iterations=N)[1] / N
			return tf.reduce_mean([
				mix.log_prob( tf.where(tf.math.is_nan(y), mix.sample(), y) )
                #把y从一个数转换为一个分布
			for _ in range(N)], 0)

		# Much slower due to cond executing both branches regardless of the conditional
		# likelihood = tf.cond(tf.reduce_any(tf.math.is_nan(y)), lambda: impute(mix, y, self.imputations), lambda: mix.log_prob(y))
		likelihood = mix.log_prob(y)
    
		return tf.reduce_mean(-likelihood) + tf.add_n([0.] + self.model.losses)#计算这两个分布的相似性


	def __call__(self, inputs):
		return self.model(inputs)


	def get_config(self):
		return self.config


	def set_config(self, config, *args, **kwargs):
		self.config = {} 
		self.update_config(config, *args, **kwargs)


	def update_config(self, config, keys=None):
		if keys is not None:
			config = {k:v for k,v in config.items() if k in keys or k not in self.config}
		
		self.config.update(config)
		for k, v in config.items():
			setattr(self, k, v)


	def save(self):
		self.model_path.mkdir(parents=True, exist_ok=True)
		store_pkl(self.model_path.joinpath('config.pkl'), self.get_config())
		self.model.save_weights(self.model_path.joinpath('checkpoint'))


	def load(self):
		self.update_config(read_pkl(self.model_path.joinpath('config.pkl')), ['scalerx', 'scalery', 'tf_random', 'np_random'])
		tf.random.set_global_generator(self.tf_random)
		if not hasattr(self, 'model'): self.build()
		self.model.load_weights(self.model_path.joinpath('checkpoint')).expect_partial()


	def get_coefs(self, output):
		prior, mu, scale = self._parse_outputs(output)
		return prior, mu, self._covariance(scale)
		#scale做了协方差

	def _parse_outputs(self, output):
		prior, mu, scale = tf.split(output, [self.n_mix, self.n_mix * self.n_targets, -1], axis=1)
		prior = tf.reshape(prior, shape=[-1, self.n_mix])
		mu    = tf.reshape(mu,    shape=[-1, self.n_mix, self.n_targets])
		scale = tf.reshape(scale, shape=[-1, self.n_mix, self.n_targets, self.n_targets])
		return prior, mu, scale


	def _covariance(self, scale):
		return tf.einsum('abij,abjk->abik', tf.transpose(scale, perm=[0,1,3,2]), scale)



	'''
	Estimate Generation
	'''
    #不同的估计方式
	def _calculate_top(self, prior, values):
		vals, idxs  = tf.nn.top_k(prior, k=1)
		idxs = tf.stack([tf.range(tf.shape(idxs)[0]), tf.reshape(idxs, [-1])], axis=-1)
		return tf.gather_nd(values, idxs)

	def _get_top_estimate(self, coefs, **kwargs):
		prior, mu, _ = coefs
		return self._calculate_top(prior, mu)

	def _get_avg_estimate(self, coefs, **kwargs):
		prior, mu, _ = coefs
		return tf.reduce_sum(mu * tf.expand_dims(prior, -1), 1)

	def _get_threshold_estimate(self, coefs, threshold=0.5):
		top_estimate = self.get_top_estimate(coefs)
		avg_estimate = self.get_avg_estimate(coefs)
		prior, _, _  = coefs
		return tf.compat.v2.where(tf.expand_dims(tf.math.greater(tf.reduce_max(prior, 1) / threshold, tf.math.sign(threshold)), -1), top_estimate, avg_estimate)


	'''
	Confidence Estimation
	'''
	def _calculate_confidence(self, sigma, level=0.9):
		# For a given confidence level probability p (0<p<1), and number of dimensions d, rho is the error bar coefficient: rho=sqrt(2)*erfinv(p ** (1/d))
		# https://faculty.ucmerced.edu/mcarreira-perpinan/papers/cs-99-03.pdf
		s, u, v = tf.linalg.svd(sigma)
		rho = 2**0.5 * tf.math.erfinv(level ** (1./self.n_targets)) 
		return tf.cast(rho, tf.float32) * 2 * s ** 0.5

	def _get_top_confidence(self, coefs, level=0.9):
		prior, mu, sigma = coefs
		top_sigma = self._calculate_top(prior, sigma)
		return self._calculate_confidence(top_sigma, level)		

	def _get_avg_confidence(self, coefs, level=0.9):
		prior, mu, sigma = coefs
		avg_estim = self.get_avg_estimate(coefs)
		avg_sigma = tf.reduce_sum(tf.expand_dims(tf.expand_dims(prior, -1), -1) * 
						(sigma + tf.matmul(tf.transpose(mu - tf.expand_dims(avg_estim, 1), (0,2,1)), 
														mu - tf.expand_dims(avg_estim, 1))), axis=1)
		return self._calculate_confidence(avg_sigma, level)		




class MixtureLayer(tf.keras.layers.Layer):

	def __init__(self, n_mix, n_targets, epsilon, **layer_kwargs):
		super(MixtureLayer, self).__init__()
		layer_kwargs.pop('activation', None)

		self.n_mix     = n_mix 
		self.n_targets = n_targets 
		self.epsilon   = tf.constant(epsilon)
		self._layer    = tf.keras.layers.Dense(self.n_outputs, **layer_kwargs)
		#前面的参数直接传过来
	@property 
	def layer_sizes(self):
		''' Sizes of the prior, mu, and (lower triangle) scale matrix outputs '''
		sizes = [1, self.n_targets, (self.n_targets * (self.n_targets + 1)) // 2]
		return self.n_mix * np.array(sizes)


	@property 
	def n_outputs(self):
		''' Total output size of the layer object '''
		return sum(self.layer_sizes)


	# @tf.function(experimental_compile=True)
	def call(self, inputs):
        #这里是前向传播
        #整个输入分为三个部分
		prior, mu, scale = tf.split(self._layer(inputs), self.layer_sizes, axis=1)

		prior = tf.nn.softmax(prior, axis=-1) + tf.constant(1e-9)
        #softmax激活，为了不为0加了个常数
		mu    = tf.stack(tf.split(mu, self.n_mix, 1), 1) 
		scale = tf.stack(tf.split(scale, self.n_mix, 1), 1) 
		scale = tfp.math.fill_triangular(scale, upper=False)
        # 变成一个上三角矩阵
		norm  = tf.linalg.diag(tf.ones((1, 1, self.n_targets)))
        # 取出来对角线的值
		sigma = tf.einsum('abij,abjk->abik', tf.transpose(scale, perm=[0,1,3,2]), scale)
        #矩阵乘法
		sigma+= self.epsilon * norm
		scale = tf.linalg.cholesky(sigma)
	#乘出来一个分布
		return tf.keras.layers.concatenate([
			tf.reshape(prior, shape=[-1, self.n_mix]),
			tf.reshape(mu,    shape=[-1, self.n_mix * self.n_targets]),
			tf.reshape(scale, shape=[-1, self.n_mix * self.n_targets ** 2]),
		])
    #三个矩阵压缩到同一个矩阵里进行输出
```

这个论文的一个特殊的地方，就是他的loss。

虽然最后输出的是只输出一个数，但是他在训练这个网络的时候，用了pdf和pdf的极大似然值来做loss

就是这里

```python
def loss(self, y, output):
    prior, mu, scale = self._parse_outputs(output) 
    dist  = getattr(tfp.distributions, self.distribution)(mu, scale)
    prob  = tfp.distributions.Categorical(probs=prior)
    mix   = tfp.distributions.MixtureSameFamily(prob, dist)

    def impute(mix, y, N):
        # summation  = tf.zeros(tf.shape(y)[0])
        # imputation = lambda i, s: [i+1, tf.add(s, mix.log_prob(tf.where(tf.math.is_nan(y), mix.sample(), y)))]
        # return tf.while_loop(lambda i, x: i < N, imputation, (0, summation), maximum_iterations=N, parallel_iterations=N)[1] / N
        return tf.reduce_mean([
            mix.log_prob( tf.where(tf.math.is_nan(y), mix.sample(), y) )
            for _ in range(N)], 0)

    # Much slower due to cond executing both branches regardless of the conditional
    # likelihood = tf.cond(tf.reduce_any(tf.math.is_nan(y)), lambda: impute(mix, y, self.imputations), lambda: mix.log_prob(y))
    likelihood = mix.log_prob(y)
    return tf.reduce_mean(-likelihood) + tf.add_n([0.] + self.model.losses)
```

返回值是tf.reduce_mean(-likelihood) + tf.add_n([0.] + self.model.losses)

impute这个函数似乎一直都没有用到

计算张量的各个维度上的元素的平均值.

而这里计算的是likelihood，计算方式是mix.log_prob(y)

mix这个类来自于tfp.distributions.MixtureSameFamily(prob, dist), prob和dist来自output

tfp这个东西来自于tensorflow_probability

接下来请见第三篇，tensorflow_probability



