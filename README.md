
# Handwritten Digit Classification using Deep Learning (MNIST)

This project implements a deep learning model using **TensorFlow and Keras** to classify handwritten digits (0–9) from the **MNIST dataset**.  
The model is trained on grayscale images and achieves high accuracy using a simple neural network architecture.

---

## 📌 Project Overview

Handwritten digit recognition is a classic problem in deep learning and computer vision.  
In this project, a **feedforward neural network** is built to recognize digits from images of size 28×28 pixels.

---

## 🧠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy

---

## 📂 Dataset

- **MNIST Dataset**
- 60,000 training images
- 10,000 testing images
- Image size: 28×28 pixels
- Pixel values normalized from 0–255 to 0–1

---

## 🏗️ Model Architecture

- Flatten Layer (28×28 → 1D)
- Dense Layer (128 neurons, ReLU activation)
- Dropout Layer (0.2)
- Output Layer (10 neurons for digits 0–9)

---

## ⚙️ How It Works

1. Load the MNIST dataset
2. Normalize the image data
3. Build a neural network model
4. Train the model using Adam optimizer
5. Evaluate accuracy on test data
6. Predict handwritten digits

---
