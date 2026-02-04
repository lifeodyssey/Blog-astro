---
title: PyTorch Basics
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
  - PyTorch
categories:
  - Learning Notes
abbrlink: fee729e7
slug: pytorch-basics
date: 2021-12-31 10:43:02
mathjax: true
copyright: true
lang: en
---

I never thought I'd start training neural networks one day.

<!-- more -->

Main reference is [this](https://github.com/zergtant/pytorch-handbook)

# PyTorch Basics

## Tensors

The basic unit in numpy is ndarray, while in PyTorch it's tensor. Tensors can be computed on GPU.

Basic operations are the same, like ones_like, zeros, size, and indexing methods are also the same.

There are five basic data types:

- 32-bit float: torch.FloatTensor (default)
- 64-bit integer: torch.LongTensor
- 32-bit integer: torch.IntTensor
- 16-bit integer: torch.ShortTensor
- 64-bit float: torch.DoubleTensor

Besides these numeric types, there are also byte and char types.

### Special Tensor Operations

Addition method 1:

```python
y = torch.rand(5, 3)
print(x + y)
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

Addition method 2:

```python
print(torch.add(x, y))
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

Providing an output tensor as argument:

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

In-place replacement:

```python
# adds x to y
y.add_(x)
print(y)
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

#### Note

Any operation ending with ``_`` will replace the original variable with the result. For example: ``x.copy_(y)``, ``x.t_()``, will all modify ``x``.

Changing dimensions:

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

If you have a tensor with only one element, use `.item()` to get the Python numeric value:

```python
x = torch.randn(1)
print(x)
print(x.item())
tensor([-0.2368])
-0.23680149018764496
```

### Conversion with NumPy

Torch Tensor and NumPy arrays share the underlying memory address, modifying one will cause the other to change.

Converting a Torch Tensor to NumPy array:

```python
a = torch.ones(5)
print(a)
tensor([1., 1., 1., 1., 1.])
```

```python
b = a.numpy()
print(b)
[1. 1. 1. 1. 1.]
```

Observe how the numpy array's value changes:

```python
a.add_(1)
print(a)
print(b)
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

Converting NumPy Array to Torch Tensor:

Use from_numpy for automatic conversion:

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

All Tensor types are CPU-based by default. CharTensor does not support conversion to NumPy.

Use the `.to` method to move Tensors to any device:

```python
# is_available function checks if cuda is available
# ``torch.device`` moves tensors to the specified device
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # create tensor directly on GPU
    x = x.to(device)                       # or use ``.to("cuda")`` to move tensor to cuda
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change the variable type
tensor([0.7632], device='cuda:0')
tensor([0.7632], dtype=torch.float64)
```

## Autograd

I didn't quite understand what this is used for.

## Neural Network Package & Optimizers

torch.nn is a modular interface specifically designed for neural networks. nn is built on top of Autograd and can be used to define and run neural networks. Here we mainly introduce some commonly used classes.

**Convention: We set torch.nn alias as nn for convenience. Besides nn, there are other naming conventions in this chapter.**

```python
# First import the relevant packages
import torch
# Import torch.nn and set alias
import torch.nn as nn
# Print the version
torch.__version__
```

```bash
'1.0.0'
```

Besides the nn alias, we also import nn.functional, which contains some commonly used functions in neural networks. These functions are characterized by having no learnable parameters (such as ReLU, pool, DropOut, etc.). These functions can be placed in the constructor or not, but it's recommended not to.

Generally, **we set nn.functional as uppercase F** for convenient calling:

```python
import torch.nn.functional as F
```

### Defining a Network

PyTorch has ready-made network models for us. Just inherit nn.Module and implement its forward method. PyTorch will automatically implement the backward function based on autograd. In the forward function, you can use any function supported by tensor, and you can also use if, for loops, print, log, and other Python syntax, written the same as standard Python.

```python
class Net(nn.Module):
    def __init__(self):
        # nn.Module subclass must execute parent class constructor in constructor
        super(Net, self).__init__()

        # Convolutional layer '1' means input image is single channel, '6' means output channels, '3' means 3*3 kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        # Linear layer, input 1350 features, output 10 features
        self.fc1   = nn.Linear(1350, 10)  # How is 1350 calculated? See the forward function below
    # Forward propagation
    def forward(self, x):
        print(x.size()) # Result: [1, 1, 32, 32]
        # Convolution -> Activation -> Pooling
        x = self.conv1(x) # According to convolution size formula, result is 30
        x = F.relu(x)
        print(x.size()) # Result: [1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2)) # Using pooling layer, result is 15
        x = F.relu(x)
        print(x.size()) # Result: [1, 6, 15, 15]
        # reshape, '-1' means adaptive
        # This flattens [1, 6, 15, 15] to [1, 1350]
        x = x.view(x.size()[0], -1)
        print(x.size()) # This is the input 1350 for fc1 layer
        x = self.fc1(x)
        return x

net = Net()
print(net)
```

```bash
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=1350, out_features=10, bias=True)
)
```

The network's learnable parameters are returned via net.parameters():

```python
for parameters in net.parameters():
    print(parameters)
```

```bash
Parameter containing:
tensor([[[[ 0.2745,  0.2594,  0.0171],
          [ 0.0429,  0.3013, -0.0208],
          [ 0.1459, -0.3223,  0.1797]]],
        ...
        [[[ 0.1691, -0.0790,  0.2617],
          [ 0.1956,  0.1477,  0.0877],
          [ 0.0538, -0.3091,  0.2030]]]], requires_grad=True)
...
```

net.named_parameters can return both learnable parameters and their names:

```python
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())
```

```bash
conv1.weight : torch.Size([6, 1, 3, 3])
conv1.bias : torch.Size([6])
fc1.weight : torch.Size([10, 1350])
fc1.bias : torch.Size([10])
```

```python
input = torch.randn(1, 1, 32, 32) # The input here corresponds to 32 in forward
out = net(input)
out.size()
```

```bash
torch.Size([1, 10])
```

Before backpropagation, all parameter gradients must be zeroed:

```python
net.zero_grad()
out.backward(torch.ones(1,10)) # Backpropagation is automatically implemented by PyTorch
```

**Note**: torch.nn only supports mini-batches, not single sample input at a time, meaning it must be a batch.

In other words, even if we input one sample, it will be batched. So all inputs will have an additional dimension. Comparing with the input above, nn defines it as 3D, but we manually added one dimension to make it 4D, where the first 1 is the batch-size.

I feel I need to review coursera instead of looking at code here.

#### Forward Propagation, Backward Propagation, Neural Networks

Found [this](https://www.techbrood.com/zh/news/ai/%E7%94%A8%E4%BA%BA%E8%AF%9D%E8%A7%A3%E9%87%8A%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E9%87%8C%E9%9D%A2%E7%9A%84%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%92%8C%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.html)

![1.png](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/202112311621806.png)

This is a neural network without hidden layers. Input on the left, output on the right. Forward is from input to output, backward is passing the Loss obtained from the back to the front output layer (like logistic).

![1552807616619421](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/202112311632948.png)

This is a neural network with hidden layers. Like before without hidden layers, we only needed to adjust one thing's parameters. Now we also need to adjust the little girl's parameters, so we use chain rule for gradient descent.

These intermediate layers and output are the legendary [activation functions](https://en.wikipedia.org/wiki/Activation_function). Without activation functions, each layer's output is a linear function of the previous layer's input, unable to fit nonlinear functions.

Input data multiplied by weight plus a bias, then apply activation function to get the neuron's output, then pass this output to the next layer's neurons. These weights and biases, plus L1, L2, batch size, etc. are hyperparameters.

### Loss Functions

## Loss Functions

In nn, PyTorch also provides commonly used loss functions. Below we use MSELoss to calculate mean squared error:

```python
y = torch.arange(0,10).view(1,10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)
# loss is a scalar, we can directly use item to get its Python numeric value
print(loss.item())
```

28.92203712463379

## Optimizers

## Optimizers

After backpropagation calculates all parameter gradients, we still need to use optimization methods to update network weights and parameters. For example, the update strategy for Stochastic Gradient Descent (SGD) is:

weight = weight - learning_rate * gradient

Most optimization methods are implemented in torch.optim, such as RMSProp, Adam, SGD, etc. Below we use SGD for a simple example:

```python
import torch.optim
```

```python
out = net(input) # Calling here will print the size of x from our forward function
criterion = nn.MSELoss()
loss = criterion(out, y)
# Create a new optimizer, SGD only needs parameters to adjust and learning rate
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
# Zero gradients first (same effect as net.zero_grad())
optimizer.zero_grad()
loss.backward()

# Update parameters
optimizer.step()
```

```bash
torch.Size([1, 1, 32, 32])
torch.Size([1, 6, 30, 30])
torch.Size([1, 6, 15, 15])
torch.Size([1, 1350])
```

This completes a full data propagation through the neural network using PyTorch.

I didn't look at the rest in detail.

## Fine Tuning

This happens to match my current needs.

For a certain task, if you don't have much training data, what to do? No worries, we first find a similar model that someone else has trained, take their ready-made trained model, swap in our own data, adjust some parameters, and train again. This is fine-tuning. The classic network models provided in PyTorch are all pre-trained by officials using the Imagenet dataset. If our training data is insufficient, these can be used as base models.

1. If the new dataset is similar to the original dataset, you can directly fine-tune the last FC layer or specify a new classifier
2. If the new dataset is small and quite different from the original dataset, you can start training from the middle of the model, only fine-tuning the last few layers
3. If the new dataset is small and quite different from the original dataset, and the above method still doesn't work, it's best to retrain, only using the pretrained model as initialization data for a new model
4. The new dataset size must be the same as the original dataset. For example, in CNN, the input image size must be the same to avoid errors
5. If dataset sizes are different, you can add convolution or pool layers before the last fc layer to make the final output match the fc layer, but this will significantly reduce accuracy, so it's not recommended
6. Different layers can have different learning rates. Generally, it's recommended that layers using original data for initialization should have a learning rate smaller than (usually 10 times smaller) the initialization learning rate. This ensures that already initialized data won't be distorted too quickly, while new layers using initialization learning rate can converge quickly.

For more details, see [here](https://github.com/zergtant/pytorch-handbook/blob/master/chapter4/4.1-fine-tuning.ipynb)

There's also some [visualization](https://github.com/zergtant/pytorch-handbook/blob/master/chapter4/4.2.2-tensorboardx.ipynb) content, will add later when needed.

