 ### Flower Classification Using CNN

This repository contains an end-to-end implementation of a Convolutional Neural Network (CNN) for classifying images of flowers. The project uses TensorFlow and Keras for model building, training, and evaluation.

---

## Table of Contents

Overview

Project Structure

Setup Instructions

Training the Model

Evaluating the Model

Using the Model

Dependencies

License

---

## Overview

This project demonstrates a simple yet effective CNN architecture for image classification. The model is trained on the "flowers" dataset and predicts the class of a given flower image.

**Key features:**

Multiple convolutional layers for feature extraction

MaxPooling layers for downsampling

Fully connected layers with Dropout for classification

Early stopping to prevent overfitting

---

## Project Structure

New-End-to-end-CNN-task/
├── flowers/              # Dataset folder containing subfolders for each flower class
├── flower_cnn.h5         # Trained CNN model
├── train_cnn.py          # Python script for training the CNN (your provided code)
├── .gitignore            # Git ignore file
├── requirements.txt      # Python dependencies
└── README.md             # This file

---

## Setup Instructions

**Clone the repository:**

git clone https://github.com/kavinkumartk/New-End-to-end-CNN-task.git
cd New-End-to-end-CNN-task

**Training the Model**

To train the CNN model, run:

python train_cnn.py

---

## 📈 Training vs Validation Accuracy

The plot represents the model performance across epochs:

- **Blue Line (Train Acc)** → Training accuracy improvement as the model learns.
- **Orange Line (Val Acc)** → Validation accuracy to monitor generalization performance.

![Model Accuracy](c202082f-e295-49eb-a64e-aa7012b6db68.png)

### Observations:
1. Training accuracy starts around **40%** and gradually improves to **80%**.
2. Validation accuracy starts around **50%** and stabilizes near **74%**.
3. After ~15 epochs, training accuracy continues to increase, but validation accuracy plateaus, indicating **possible overfitting**.

---

## Using the Model with Gradio

A Gradio web interface is included for easy inference:

Run the app:
python app.py

---

## Dependencies

Python 3.x

TensorFlow

Keras

Matplotlib

NumPy

---

## 📌 Future Improvements

Implement cross-validation to better evaluate performance.

Use hyperparameter tuning for better optimization.

Experiment with advanced models like CNN, RNN, or Transformers.

---

**Author : KAVINKUMAR T**

