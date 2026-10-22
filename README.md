Mulberry Leaf Classification Using EfficientNetB0 and Curriculum Learning

1. Project Overview

This project focuses on classifying mulberry leaf varieties using deep learning. It uses the EfficientNetB0 convolutional neural network as a feature extractor and investigates whether curriculum learning can improve classification performance compared with random training.

The objective is to identify the variety of a mulberry leaf from an input image. This project performs variety classification, not disease detection.

2. Objectives

* Classify mulberry leaves into 10 different varieties.
* Develop a baseline classification model using EfficientNetB0 with ImageNet-pretrained weights.
* Investigate curriculum learning as an alternative to random training.
* Compare the strategies using test accuracy and Macro-F1 score.
* Analyze the results, limitations, and potential improvements.

3. Dataset Description

The dataset contains 5,262 mulberry leaf images belonging to 10 varieties.

Mulberry Leaf Classes

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

Dataset Split

Dataset	Number of Images
Training	2,977
Validation	742
Testing	1,543
Total	5,262

Input images are resized to 224 × 224 pixels and processed as RGB images.

4. Methodology

The project investigates two experimental approaches.

Experiment A: Baseline EfficientNetB0

* Uses EfficientNetB0 with ImageNet-pretrained weights.
* Keeps the pretrained backbone frozen.
* Applies Global Average Pooling.
* Adds a fully connected layer with 128 ReLU units.
* Uses Dropout with a rate of 0.3.
* Uses a final Softmax layer with 10 output units.
* Applies image augmentation during training.
* Uses Adam optimization with a learning rate of 0.001.
* Uses categorical cross-entropy as the loss function.
* Uses early stopping based on validation loss.

Experiment B: Training Strategy Comparison

This experiment compares three training strategies using extracted EfficientNetB0 feature vectors.

The feature vectors have 1,280 dimensions. A class-centroid cosine-similarity margin heuristic is used to estimate sample difficulty.

The strategies are:

* Random Training: Training samples are selected without curriculum-based ordering.
* Curriculum Learning: Training begins with easier samples and progressively introduces more difficult samples.
* Curriculum-Shuffled: Uses the curriculum-based sample selection approach with shuffling.

The curriculum schedule begins with the easiest 30% of samples and progressively expands to the full training set by epoch 11.

Each strategy is evaluated using three random seeds: 0, 1, and 2.

5. Model Architecture

The baseline model consists of:

1. Input image of size 224 × 224 × 3.
2. Pretrained EfficientNetB0 backbone.
3. Global Average Pooling layer.
4. Dense layer with 128 ReLU units.
5. Dropout layer with a rate of 0.3.
6. Dense Softmax output layer with 10 units.

The baseline model has 4,214,829 total parameters, of which 165,258 are trainable, according to the reported model summary.

6. Experimental Results

Experiment A: Baseline EfficientNetB0

Metric	Result
Validation Accuracy	77.76%
Validation Macro-F1	77.75%
Test Accuracy	77.64%
Test Macro-F1	76.84%

Experiment B: Training Strategy Comparison

The following values are the reported mean test results across three random seeds.

Training Strategy	Mean Test Accuracy	Mean Test Macro-F1
Random Training	80.62%	80.55%
Curriculum Learning	77.19%	78.34%
Curriculum-Shuffled	80.13%	80.35%

Test Accuracy by Random Seed

Random Seed	Random Training	Curriculum Learning	Curriculum-Shuffled
0	81.59%	77.64%	79.71%
1	79.07%	76.86%	79.59%
2	81.21%	77.06%	81.08%

Results Interpretation

In the training strategy comparison, Random Training achieved the highest mean test accuracy and mean test Macro-F1 among the three strategies.

Curriculum Learning did not outperform Random Training in these experiments. Curriculum-Shuffled performed closer to Random Training.

These findings suggest that curriculum learning did not provide a performance improvement under the experimental conditions used in this project.

Important: The baseline experiment and the training strategy comparison use different training pipelines. Their results should not be treated as a perfectly controlled, direct comparison.

7. Evaluation Metrics

Accuracy

Accuracy measures the proportion of correctly classified images out of all evaluated images.

Macro-F1 Score

Macro-F1 calculates the F1 score separately for each class and then takes the unweighted average. It helps evaluate performance across classes, including classes with fewer samples.

Both metrics are used to assess the classification performance of the models.

8. Limitations

* The dataset may contain an imbalance in the number of images per variety.
* Sample difficulty is estimated using a class-centroid cosine-similarity margin heuristic.
* The training strategy comparison uses only three random seeds.
* The baseline and feature-extraction experiments use different pipelines, limiting direct comparison.
* The reported experiments do not establish how well the model generalizes to an independent external dataset.
* The baseline experiment uses a frozen EfficientNetB0 backbone rather than fine-tuning the entire network.

9. Future Improvements

* Expand the dataset with more diverse images for each variety.
* Investigate methods to address class imbalance.
* Evaluate alternative sample-difficulty measures and curriculum schedules.
* Run experiments with more random seeds for more reliable comparisons.
* Analyze a confusion matrix and incorrectly classified images to identify challenging varieties.
* Evaluate the model on an independent dataset.
* Investigate fine-tuning EfficientNetB0 as a separate experiment.
* Compare the training strategies under a more tightly controlled experimental setup.

10. Technologies Used

* Python
* TensorFlow
* Keras
* EfficientNetB0
* NumPy
* Scikit-learn
* Matplotlib

11. How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Open the project notebook in Jupyter Notebook or Google Colab.
4. Update the dataset path to match your environment.
5. Run the notebook cells in order.

The original notebook may use machine-specific dataset paths, so these must be updated before running the experiments.

12. Reproducibility

The training strategy comparison uses random seeds 0, 1, and 2. Results may vary depending on the software environment, hardware, dataset organization, and training configuration.

For reproducibility, use the same dataset split, preprocessing steps, model configuration, and training settings as those used in the reported experiments.
