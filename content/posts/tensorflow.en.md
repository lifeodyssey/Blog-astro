---
title: TensorFlow Basics
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
  - tensorflow
categories:
  - Learning Notes
abbrlink: fd169b9d
slug: tensorflow-basics
date: 2022-01-04 10:43:02
mathjax: true
copyright: true
lang: en
---

Day two of starting to train neural networks.

<!-- more -->

# A Simple Neural Network Building Process

```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

Output:

```bash
Epoch 1/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2962 - accuracy: 0.9155
Epoch 2/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1420 - accuracy: 0.9581
Epoch 3/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1064 - accuracy: 0.9672
Epoch 4/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0885 - accuracy: 0.9730
Epoch 5/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0749 - accuracy: 0.9765
313/313 - 0s - loss: 0.0748 - accuracy: 0.9778
[0.07484959065914154, 0.9778000116348267]
```

Let's look at the structure of this simple neural network.

`x_train, x_test = x_train / 255.0, x_test / 255.0` is because the MNIST dataset is 0-255, so normalization is applied.

`tf.keras.models.Sequential` is a computational connector, meaning this network is connected by the following steps and computed in order.

The first layer `tf.keras.layers.Flatten` converts the image format from a 2D array (28 x 28 pixels) to a 1D array (28 x 28 = 784 pixels). Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.

After flattening the pixels, the network includes a sequence of two `tf.keras.layers.Dense` layers. These are densely connected or fully connected neural layers. The first `Dense` layer has 128 nodes (or neurons). The second (and last) layer returns a logits array of length 10. Each node contains a score indicating which of the 10 classes the current image belongs to.

The Dropout here is a regularization trick to prevent overfitting. Dropout means: randomly ignoring some neurons during each training iteration - these neurons are "dropped out". In other words, these neurons' activation effects on downstream are ignored during forward propagation, and weights are not updated during backpropagation.

The so-called "learning" of neural networks means making each neuron's weights fit the required characteristics. Different neuron combinations can distinguish certain features of data. Each neuron's neighbors depend on features composed of neighbor behavior. If over-dependent, it causes overfitting. If some neurons are randomly removed each time, the remaining neurons need to compensate for the missing neurons' functions, and the entire network becomes a collection of many independent networks (different solutions to the same problem).

model.compile requires three things:

- *Loss function* - Measures model accuracy during training. You want to minimize this function to "guide" the model in the right direction.
- *Optimizer* - Determines how the model updates based on data and its loss function.
- *Metrics* - Used to monitor training and testing steps. The example uses *accuracy*, the ratio of correctly classified images.

Loss function and metrics are two separate things due to different purposes.

The loss function exists to be minimized during parameter optimization. Its purpose is to optimize model parameters - when this value is minimum, the model is best. It generally requires derivatives or gradients. Metrics are the evaluations we care about.

For example, in regression, MSE can be both a loss function and a metric, but R² may not work as a loss function because larger R² means better model.

# Regression

Most of the process is the same as before. Example from [here](https://tensorflow.google.cn/tutorials/keras/regression).

```python
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
```

Since the output is a single value, the last layer has only one neuron.

[callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) are functions executed at the end of each epoch.

validation sets up cross-validation.

The history object can visualize model training progress.

# Overfitting and Underfitting

The simplest way to prevent overfitting is to start with a small model - one with few learnable parameters (determined by number of layers and units per layer). In deep learning, the number of learnable parameters is often called the model's "capacity".

Intuitively, a model with more parameters will have more "memorization capacity" and can easily learn a perfect dictionary-like mapping between training samples and targets - a mapping with no generalization ability, useless for predictions on unseen data.

Always remember: deep learning models are good at fitting training data, but the real challenge is generalization, not fitting.

## Training Process

Generally, smaller learning rates work better. We can use [`tf.keras.optimizers.schedules`](https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/schedules) to reduce learning rate over time.

## Saving and Restoring Models

Wow, they save in HDF5 format - no wonder OC-SMART uses this format.

During training, use tf.keras.callbacks.ModelCheckpoint to save models.

As long as two models share the same architecture, you can share weights between them. When restoring a model from weights only, create a model with the same architecture as the original, then set its weights.

# Hyperparameter Tuning

Keras Tuner is a library that helps you select the best hyperparameter set for your TensorFlow program. Selecting the right hyperparameter set for your ML application is called *hyperparameter tuning* or *hypertuning*.

Hyperparameters are variables that control the training process and ML model topology. These variables remain constant during training and directly affect ML program performance.

# Fine-tuning

In this tutorial, you'll learn how to use transfer learning to classify cat and dog images with a pretrained network.

A pretrained model is a saved network previously trained on a large dataset (typically a large image classification task). You can use the pretrained model as-is or use transfer learning to customize it for a given task.

# Back to My Own Problem

I need to use the H5 format, so first I need to recreate a model myself.

I also manually tested the model structure. The next question is: how to assign values to Weight and Bias.

It's so simple... as long as you're willing to check the source code, nothing is a problem.

```python
Fmodel.add(Dense(100, input_shape=(7,),activation='tanh',use_bias=True))
Fmodel.layers[0].set_weights([np.array(f['Weights']['Layer1']).T,np.array(f['Bias']['Layer1']).flatten()])
```

Finally got it working. Now just need to do fine-tuning.

