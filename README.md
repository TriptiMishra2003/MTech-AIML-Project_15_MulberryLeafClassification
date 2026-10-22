# MTech-AIML-Project_15_MulberryLeafClassification

# Mulberry Leaf Classification Using EfficientNetB0

## 1. Project Overview

This project focuses on classifying mulberry leaf images into
10 different classes using deep learning and transfer learning.

The project uses the pretrained EfficientNetB0 architecture
with additional classification layers for mulberry leaf
image classification.

## 2. Objectives

- Explore and understand the mulberry leaf dataset.
- Analyze the distribution of images across classes.
- Preprocess images and create training and validation sets.
- Train an EfficientNetB0-based classification model.
- Monitor training and validation accuracy and loss.
- Evaluate the trained model and analyze its performance.

## 3. Dataset

The dataset contains images belonging to 10 mulberry leaf classes.

The dataset is organized into training and testing directories.

See `data/README.md` for dataset organization and access
instructions.

## 4. Methodology

1. Dataset exploration and class-wise image counting.
2. Visualization of class-wise image distribution.
3. Image resizing to 224 x 224 pixels.
4. Training and validation split.
5. Data augmentation for training images.
6. Transfer learning using pretrained EfficientNetB0.
7. Model training using categorical cross-entropy loss.
8. Evaluation using accuracy and loss.

## 5. Project Structure
- `requirements.txt`: Required Python packages.
- `data/`: Dataset documentation and sample data, if applicable.
- `notebooks/`: Jupyter Notebook experiments.
- `src/`: Reusable preprocessing, model and evaluation code.
- `results/`: Results, comparison tables and figures.
- `presentation/`: Project presentation.


## 6. Model

- Architecture: EfficientNetB0
- Input image size: 224 x 224 x 3
- Number of classes: 10
- Classification activation: Softmax
- Loss function: Categorical cross-entropy
- Optimizer: Adam

## 7. Results

Training and validation accuracy, loss curves and evaluation
results will be documented here after the experiments are run.

Best validation accuracy:

Test accuracy: 




