Mulberry Leaf Classification Using EfficientNetB0 and Curriculum Learning

1. Project Overview

This project focuses on classifying images of mulberry leaves into 10 different varieties using deep learning and transfer learning with EfficientNetB0.

It also investigates whether Curriculum Learning improves classification performance compared with Random Training and Curriculum-Shuffled Training.

Project task: Mulberry leaf variety classification, not disease detection.

2. Objectives

* Explore and preprocess the mulberry leaf dataset.
* Develop an EfficientNetB0-based image classification model.
* Evaluate classification performance using accuracy and Macro-F1.
* Compare Random Training, Curriculum Learning, and Curriculum-Shuffled Training.
* Analyse experimental results, limitations, and potential improvements.

3. Dataset

The dataset contains 5,262 images across 10 mulberry leaf varieties.

Dataset Split	Number of Images
Training	2,977
Validation	742
Testing	1,543
Total	5,262

Image size: 224 × 224 pixels, RGB.

Classes

1. ChiangMai60
2. RedKing
3. WhiteKing
4. BlackOodTurkey
5. TaiwanStraberry
6. BlackAustralia
7. Buriram60
8. Kamphaengsaeng42
9. TaiwanMeacho
10. ChiangMaiBuriram60

The validation set is created by splitting the original training directory. The separate test set is used for final evaluation.

The test set contains unequal numbers of images per class, making class-wise evaluation important.

Note: The dataset is stored locally. Update the dataset paths in the notebook to match your environment.

4. Methodology

The project consists of two experiments.

Experiment A: EfficientNetB0 Baseline

1. Load EfficientNetB0 with ImageNet-pretrained weights.
2. Freeze the pretrained EfficientNetB0 backbone.
3. Resize input images to 224 × 224 pixels.
4. Apply data augmentation to training images.
5. Add Global Average Pooling, a Dense layer with 128 neurons and ReLU, Dropout (0.3), and a final Dense layer with 10 Softmax outputs.
6. Train the classification head using the Adam optimizer and categorical cross-entropy.
7. Monitor validation loss using Early Stopping.

Experiment B: Feature Extraction and Training Strategy Comparison

EfficientNetB0 extracts a 1,280-dimensional feature vector from each image. Three training strategies are compared:

* Random Training: Training examples are presented in random order.
* Curriculum Learning: Training begins with easier examples and gradually introduces harder examples.
* Curriculum-Shuffled: Uses progressively larger curriculum-selected subsets, with examples shuffled within each subset.

Sample difficulty is estimated using cosine-similarity margins between image features and class centroids.

The three strategies are evaluated using random seeds 0, 1, and 2.

Important: The baseline and feature-extraction experiments use different training pipelines. Their results are not a perfectly controlled, direct comparison.

5. Model Architecture

Component	Configuration
Base model	EfficientNetB0
Pretrained weights	ImageNet
Input shape	224 × 224 × 3
Number of classes	10
Pooling	Global Average Pooling
Hidden layer	Dense, 128 neurons, ReLU
Regularisation	Dropout, 0.3
Output layer	Dense, 10 neurons, Softmax
Optimizer	Adam
Baseline loss function	Categorical cross-entropy
Baseline batch size	32

Total parameters: 4,214,829
Trainable parameters: 165,258

The baseline freezes the EfficientNetB0 backbone and trains the classification head.

6. Experimental Results

Experiment A: EfficientNetB0 Baseline

Metric	Result
Validation Accuracy	77.76%
Validation Macro-F1	77.75%
Test Accuracy	77.64%
Test Macro-F1	76.84%

The baseline validation classification report showed:

* Highest class F1: Kamphaengsaeng42 — 89.31%.
* Lowest class F1: TaiwanMeacho — 65.82%.

These are validation results, not test results.

Experiment B: Training Strategy Comparison

The following results are the mean across three random seeds.

Training Strategy	Mean Test Accuracy	Mean Test Macro-F1
Random Training	80.62%	80.55%
Curriculum Learning	77.19%	78.34%
Curriculum-Shuffled	80.13%	80.35%

Test Accuracy by Random Seed

Training Strategy	Seed 0	Seed 1	Seed 2
Random Training	81.59%	79.07%	81.21%
Curriculum Learning	77.64%	76.86%	77.06%
Curriculum-Shuffled	79.71%	79.59%	81.08%

Key Finding

Random Training achieved the highest mean test accuracy and Macro-F1 among the three feature-extraction strategies.

Curriculum-Shuffled performed similarly to Random Training, while Curriculum Learning achieved lower results in these experiments.

These findings apply to the tested setup and do not establish that Curriculum Learning is generally ineffective.

7. Evaluation Metrics

* Accuracy: The proportion of predictions that are correct.
* Macro-F1: The average of the F1 scores calculated separately for each class, giving every class equal weight.
* Validation metrics: Used to monitor model performance during development.
* Test metrics: Used to evaluate the trained model on the held-out test set.

Using both accuracy and Macro-F1 helps assess overall performance and differences in performance across classes.

8. Limitations

* The test set contains unequal numbers of images across classes.
* The sample-difficulty score is a heuristic based on feature-centroid similarity.
* The training strategy comparison uses three random seeds.
* The baseline and feature-extraction experiments use different training pipelines.
* Generalisation to independently collected images has not been established.
* The baseline uses a frozen EfficientNetB0 backbone; fine-tuning was not evaluated in this baseline.

9. Future Improvements

* Increase dataset diversity and improve class balance.
* Analyse confusion matrices and misclassified images.
* Investigate alternative difficulty measures and curriculum schedules.
* Conduct more repeated experiments.
* Evaluate the model on an independently collected dataset.
* Compare frozen-backbone training with a controlled fine-tuning experiment.
* Document the preprocessing pipeline, class mapping, model configuration, and evaluation outputs for reproducibility.

10. Project Structure

The following is a suggested structure. Adjust it to match the actual files in your repository.

MulberryLeafClassification/
├── data/
│   └── README.md
├── notebooks/
│   └── MAI_Assignment_Final.ipynb
├── src/
├── results/
├── presentation/
├── requirements.txt
└── README.md

11. How to Run

1. Clone or download the repository.
2. Install the dependencies listed in requirements.txt.
3. Open MAI_Assignment_Final.ipynb in Jupyter Notebook or Google Colab.
4. Update the training and testing dataset paths.
5. Run the notebook cells in order.
6. Review the generated evaluation metrics and results.

Use dependency versions compatible with the notebook environment. The notebook reports TensorFlow 2.11.0; verify the package versions in requirements.txt before reproducing the experiments.

12. Reproducibility

The reported results correspond to the experiments recorded in the notebook. Re-running the experiments may produce slightly different results depending on random seeds, software versions, and the execution environment.

For reproducibility, keep the dataset organisation, class mapping, preprocessing steps, model configuration, and evaluation procedure consistent.
