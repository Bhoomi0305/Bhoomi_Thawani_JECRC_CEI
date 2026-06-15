# CIFAR-10 Image Classification using ANN and CNN

## Overview

This project demonstrates image classification on the CIFAR-10 dataset using two different deep learning architectures:

- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)

The objective is to compare the performance of ANN and CNN and analyze how architectural choices and training strategies impact classification accuracy.

## Dataset

The CIFAR-10 dataset consists of:

- 60,000 color images
- 10 image categories
- Image size: 32 × 32 pixels

## Project Workflow

### Data Preparation

- Loaded CIFAR-10 dataset using TensorFlow
- Normalized pixel values to the range [0,1]
- Flattened images for ANN input

### ANN Architecture

- Dense Layer (1024 neurons)
- Dropout Layer
- Dense Layer (512 neurons)
- Dropout Layer
- Dense Layer (256 neurons)
- Output Layer (10 classes)

### CNN Architecture

- Conv2D (32 → 64 → 128)
- Batch Normalization
- Max Pooling
- Dense Layer
- Dropout
- Softmax Output

## Training Enhancements

- Increased epochs from 10 to 20
- Early Stopping
- Data Augmentation
- Dropout Regularization
- Batch Normalization

## Results

| Model                   | Test Accuracy |
| ----------------------- | ------------- |
| ANN                     | 38.91%        |
| CNN                     | 72.11%        |
| CNN + Data Augmentation | 48.52%        |

## Conclusion

CNN significantly outperformed ANN due to its ability to preserve spatial information and learn hierarchical image features.
