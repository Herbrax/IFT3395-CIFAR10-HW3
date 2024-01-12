# Homework 3  - Pierre-Antoine Bernard & Simo Hakim - Course IFT3395/6390A - Université de Montréal

Before running the project, make sure the following packages are installed:

- NumPy
- Python 3.x
- PyTorch
- Torchvision

This Python code defines a `Trainer` class for training machine learning models, specifically Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN), using the PyTorch library. It is designed to work with the CIFAR-10 dataset.

Key components of the code:

1. **Network Configuration:**
   - `NetworkConfiguration` class: Defines network parameters like number of channels, kernel sizes, strides, and dense layer configurations.

2. **Trainer Class:**
   - Initializes network parameters, loads the dataset, and sets up the training and test data loaders.
   - Supports two types of networks: MLP and CNN. The network type is chosen based on the `network_type` parameter.
   - Includes methods to create the MLP (`create_mlp`) and CNN (`create_cnn`) using the specified configurations.
   - Defines a method `create_activation_function` to select the activation function (ReLU, Tanh, or Sigmoid).
   - Implements a training loop (`train_loop`) that iterates over epochs, updating weights using backpropagation and logging training and validation metrics like loss and mean absolute error (MAE).
   - Includes an evaluation loop (`evaluation_loop`) to compute loss and MAE on the test dataset.
   - Provides a method `evaluate` to compute loss and MAE on a given input tensor.

3. **Data Handling:**
   - The `load_dataset` method loads the CIFAR-10 dataset and applies transformations such as normalization.

4. **Random Seed Initialization:**
   - Sets a fixed random seed for NumPy, PyTorch, and the random module for reproducibility.

This code is structured to facilitate the training and evaluation of simple MLP and CNN models on image data, providing a framework for experimenting with different network architectures and parameters.
