Learning MXNet Notebooks
========================

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet_logo_2.png" title="MXNet-logo"
width="350" />

### 1) Installation and First Steps with MXNet

notebook: ([nbviewer](https://github.com/ccarpenterg/LearningMXNet/blob/master/01_installation_first_steps_mxnet.ipynb)) ([github](https://github.com/ccarpenterg/LearningMXNet/blob/master/01_installation_first_steps_mxnet.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningMXNet/blob/master/01_installation_first_steps_mxnet.ipynb))

### 2) Getting Started with MXNet: Training a Neural Network on MNIST

In the second notebook we train an artificial neural network to classify handwritten digits. We will be using the nn.Sequential module to build our neural network. We train our NN on the MNIST dataset, which is the "hello world" of Machine learning and Deep learning algorithms:

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" title="MNIST" width="375" />

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningMXNet/blob/master/02_getting_started_with_mxnet.ipynb)) ([github](https://github.com/ccarpenterg/LearningMXNet/blob/master/02_getting_started_with_mxnet.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningMXNet/blob/master/02_getting_started_with_mxnet.ipynb))

### 3) Introduction to Convolutional Neural Networks and Deep Learning

In this notebook we introduce the convolutional neural network, one of the most exciting developments in computer vision of the last decade. Convolutional neural networks, convnets or CNNs for short, are deep learning neural networks that automatically extract features from images which then are fed to a dense classifier:

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/800px-Typical_cnn.png" 
title="CNN" width="500" />

Using MXNet's gluon package we design a convnet from zero and train on the MNIST dataset.

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningMXNet/blob/master/03_introduction_to_convnets_with_mxnet.ipynb)) ([github](https://github.com/ccarpenterg/LearningMXNet/blob/master/03_introduction_to_convnets_with_mxnet.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningMXNet/blob/master/03_introduction_to_convnets_with_mxnet.ipynb))

### 4) Plotting Accuracy and Loss for CNNs with MXNet

Part of the work that involves designing and training deep neural networks, consists in plotting the various parameters and metrics generated in the process of training. In this notebook we will design and train our Convnet from scratch, and will plot the training vs. test accuracy, and the training vs. test loss.

These are very important metrics, since they will show us how well is doing our neural network.

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningMXNet/blob/master/04_plotting_accuracy_loss_convnet.ipynb)) ([github](https://github.com/ccarpenterg/LearningMXNet/blob/master/04_plotting_accuracy_loss_convnet.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningMXNet/blob/master/04_plotting_accuracy_loss_convnet.ipynb))

### 5) CIFAR-10: A More Challenging Dataset for CNNs

So far we have trained our neural networks on the MNIST dataset, and have achieved high acurracy rates for both the training and test datasets. Now we train our Convnet on the CIFAR-10 dataset, which contains 60,000 images of 32x32 pixels in color (3 channels) divided in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

<img src="https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png" title="CIFAR-10" width="295" />

As we'll see in this notebook, the CIFAR-10 dataset will prove particularly challenging for our very basic Convnet, and from this point we'll start exploring the world of pretrained neural networks.

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningMXNet/blob/master/05_cifar_10_challenging_convnets.ipynb)) ([github](https://github.com/ccarpenterg/LearningMXNet/blob/master/05_cifar_10_challenging_convnets.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningMXNet/blob/master/05_cifar_10_challenging_convnets.ipynb))

### 6) Pretrained Convolutional Neural Networks (Transfer Learning)

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningMXNet/blob/master/06_pretrained_convnets_and_transfer_learning.ipynb)) ([github](https://github.com/ccarpenterg/LearningMXNet/blob/master/06_pretrained_convnets_and_transfer_learning.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningMXNet/blob/master/06_pretrained_convnets_and_transfer_learning.ipynb))
