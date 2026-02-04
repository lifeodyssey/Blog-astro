---
title: Keras Tuner & Transfer Learning
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
  - tensorflow
categories:
  - Learning Notes
abbrlink: '56104174'
slug: keras-tuner-transfer-learning
date: 2022-02-24 10:41:27
lang: en
mathjax:
copyright:
---

It looks like this tool can speed up my model training process.

<!-- more -->

# Keras Tuner

This requires a separate installation, Python 3.6+ and TensorFlow 2.0+

```bash
pip install keras-tuner --upgrade
```

## A Quick Start Example

```python
from tensorflow import keras
from tensorflow.keras import layers


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
    )
    return model
```

Then define a tuner

```python
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)
```

Then just start it

```python
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
```

## Let's Look at the Details

Actually, the above example is not quite what I need to do.

What I need is to tune model parameters, not hyperparameters.

If hyperparameters are not needed, then what needs to be done is probably just re-fitting, which is commonly known as transfer learning.

# Transfer Learning

I already wrote a bit about this in the [previous post](https://lifeodyssey.github.io/posts/fd169b9d.html).

After reviewing it again, I found it's really quite simple.

You just need to freeze the parameters that need to be trained, and then call fit again.
