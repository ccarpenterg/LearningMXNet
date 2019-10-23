Learning MXNet Notebooks
========================

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet_logo_2.png" title="MXNet-logo"
width="350" />

### 1) Installation and First Steps with MXNet

### 2) Getting Started with MXNet: Training a Neural Network on MNIST

In the second notebook we train an artificial neural network to classify handwritten digits. We will be using the nn.Sequential module to build our neural network. We train our NN on the MNIST dataset, which is the "hello world" of Machine learning and Deep learning algorithms:

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" title="MNIST" width="375" />

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningMXNet/blob/master/02_getting_started_with_mxnet.ipynb)) ([github](https://github.com/ccarpenterg/LearningMXNet/blob/master/02_getting_started_with_mxnet.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningMXNet/blob/master/02_getting_started_with_mxnet.ipynb))

### 3) Introduction to Convolutional Neural Networks and Deep Learning

In this notebook we introduce the convolutional neural network, one of the most exciting developments in computer vision of the last decade. Convolutional neural networks, convnets or CNNs for short, are deep learning neural networks that automatically extract features from images which then are fed to a dense classifier:

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/800px-Typical_cnn.png" 
title="CNN" width="500" />

Using MXNet's gluon package we design a convnet from zero and train on the MNIST dataset.
