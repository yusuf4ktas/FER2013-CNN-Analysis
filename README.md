# Emotion Recognition From Facial Images With CNNs

## Project Overview

This project explores Facial Expression Recognition (FER) using Convolutional Neural Networks (CNNs) applied to the FER2013 dataset. The primary objective was to evaluate and compare the effectiveness of three different CNN-based models: a Custom CNN, MobileNetV2 with head-only training, and a fine-tuned MobileNetV2.

Through rigorous experimentation involving standard preprocessing and data augmentation techniques, each model was trained on a dataset characterized by notable class imbalance across seven emotional categories.

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation and Significance](#motivation-and-significance)
- [Benchmark Dataset: FER2013](#benchmark-dataset-fer2013)
- [Related Work](#related-work)
- [Research Objectives](#research-objectives)
- [Materials and Methods](#materials-and-methods)
  - [Dataset Description](#dataset-description)
  - [Preprocessing and Augmentation](#preprocessing-and-augmentation)
  - [Model Architectures](#model-architectures)
  - [Loss Function and Optimization](#loss-function-and-optimization)
  - [Evaluation Metrics and Confusion Matrices](#evaluation-metrics-and-confusion-matrices)
- [Experimental Results](#experimental-results)
  - [Training Configuration](#training-configuration)
  - [Accuracy Comparison](#accuracy-comparison)
  - [Training Behavior](#training-behavior)
  - [Prediction Samples](#prediction-samples)
- [Conclusions](#conclusions)
  - [Future Directions](#future-directions)
- [References](#references)

## Motivation and Significance

Emotions play a crucial role in human communication, and the ability of machines to interpret facial expressions enables more intuitive and responsive systems. This capability is essential in domains such as mental health monitoring, virtual assistants, and driver alertness systems. However, achieving high-performance FER remains challenging due to the variability in facial features, lighting conditions, and expressions across individuals.

## Benchmark Dataset: FER2013

The FER2013 dataset is one of the most widely used benchmarks for evaluating FER models. It comprises 35,887 grayscale images labeled across seven emotions: angry, disgusted, fearful, happy, sad, surprised, and neutral. Each image is standardized to a 48x48 resolution to simulate real-world conditions with varying poses and lighting. The dataset is partitioned into 28,709 training, 3,589 public test, and 3,589 private test images, enabling comprehensive performance evaluation.

## Related Work

Several notable CNN architectures have been applied to FER2013. VGGNet-based models have achieved high accuracy rates up to 73.28%. ResNet variants have also been employed, demonstrating better long-term accuracy due to their deep residual connections. Recent studies have proposed lightweight architectures like SpinalNet-integrated CNNs and MobileNetNext models, which aim to balance accuracy and efficiency, making them suitable for edge computing and real-time applications.

## Research Objectives

In this project, the goal was to investigate the practical application of CNN architectures for FER on the FER2013 dataset. This involved implementing a Custom CNN from scratch and comparing its performance to two transfer learning approaches using MobileNetV2: head-only training and fine-tuning. The objective was to understand how model complexity, transfer learning depth, and training strategies affect performance, especially under conditions of class imbalance and limited image resolution.

## Materials and Methods

### Dataset Description

The FER2013 dataset, a widely accepted benchmark, was used for this project. It contains 35,887 grayscale facial images at 48x48 pixels, categorized into seven emotion classes: angry, disgust, fear, happy, sad, surprise, and neutral. The dataset is divided into 28,709 training images and 7,178 test images.

The class distribution in the training set is as follows:
* Happy: 7,215 
* Neutral: 4,965 
* Sad: 4,830 
* Fear: 4,097 
* Angry: 3,995 
* Surprise: 3,171 
* Disgust: 436

<br>
<p align="center">
  <img src="images/class_distribution.png" alt="Class Distribution in Training Data" width="600"/>
  <br>
  <em>Figure 1. Class distribution across training and test sets</em>
</p>
<br>

### Preprocessing and Augmentation

All pixel values were normalized to the $[0, 1]$ range. Data augmentation techniques such as horizontal flipping, slight rotation, and zooming were applied to improve generalization and reduce overfitting.

### Model Architectures

Three different models were designed and trained:
* **Custom CNN**: A sequential convolutional network built from scratch, optimized with dropout and batch normalization.
* **MobileNetV2 (Head Only)**: Uses a pretrained MobileNetV2 with a custom classification head, freezing the base layers.
* **MobileNetV2 (Fine-Tuned)**: A fully trainable MobileNetV2 model fine-tuned on the FER2013 dataset.

#### Custom CNN Training Curves

<br>
<p align="center">
  <img src="images/custom_cnn_accuracy_loss.png" alt="Custom CNN Accuracy/Loss" width="800"/>
  <br>
  <em>Figure 2. Training accuracy/loss of Custom CNN</em>
</p>
<br>

#### MobileNetV2 (Head Only) Training Curves

<br>
<p align="center">
  <img src="images/mobilenetv2_head_only_accuracy_loss.png" alt="MobileNetV2 (Head Only) Accuracy/Loss" width="800"/>
  <br>
  <em>Figure 3. Training accuracy/loss of MobileNetV2 model</em>
</p>
<br>

#### MobileNetV2 (Fine-Tuned) Training Curves

<br>
<p align="center">
  <img src="images/mobilenetv2_fine_tuned_accuracy_loss.png" alt="MobileNetV2 (Fine-Tuned) Accuracy/Loss" width="800"/>
  <br>
  <em>Figure 4. Training accuracy/loss of MobileNetV2 fine-tuned model</em>
</p>
<br>

### Loss Function and Optimization

To address the significant class imbalance, particularly the low number of disgust images, the Focal Loss function was used instead of standard categorical cross-entropy. Focal Loss dynamically scales the loss for hard-to-classify examples:

$$ FL(p_t) = -\alpha_t(1 - p_t)^\gamma \log(p_t) $$

where:
* $p_t$ is the model’s estimated probability for the true class 
* $\alpha_t$ (alpha) balances class importance 
* $\gamma$ (gamma) is a focusing parameter (typically set to 2)

The Adam optimizer was used with a learning rate of $1e^{-3}$, a batch size of 64, and models were trained for 120 epochs with early stopping based on validation loss.

### Evaluation Metrics and Confusion Matrices

All models were evaluated based on accuracy and confusion matrices. Training histories were visualized to monitor convergence and identify overfitting.

The Custom CNN exhibited superior balance across classes, especially for "fear" and "sad" emotions. However, "disgust" remained the most difficult emotion to classify across all models due to limited data representation. MobileNetV2 (Head Only) showed the weakest results, while fine-tuning MobileNetV2 significantly improved accuracy, particularly for "surprise" and "angry".

#### Confusion Matrix - Custom CNN

<br>
<p align="center">
  <img src="images/confusion_matrix_custom_cnn.png" alt="Confusion Matrix - Custom CNN" width="600"/>
  <br>
  <em>Figure 5. Confusion Matrix of Custom CNN</em>
</p>
<br>

#### Confusion Matrix - MobileNetV2 (Head Only)

<br>
<p align="center">
  <img src="images/confusion_matrix_mobilenetv2_head_only.png" alt="Confusion Matrix - MobileNetV2 (Head Only)" width="600"/>
  <br>
  <em>Figure 6. Confusion Matrix of MultiNet without fine-tuning</em>
</p>
<br>

#### Confusion Matrix - MobileNetV2 (Fine-Tuned)

<br>
<p align="center">
  <img src="images/confusion_matrix_mobilenetv2_fine_tuned.png" alt="Confusion Matrix - MobileNetV2 (Fine-Tuned)" width="600"/>
  <br>
  <em>Figure 7. Confusion Matrix of MultiNet fine-tuned version</em>
</p>
<br>

## Experimental Results

### Training Configuration

All models were trained with consistent parameters for fair comparison:
* **Batch Size**: 64 
* **Epochs**: 120 
* **Optimizer**: Adam 
* **Learning Rate**: $1e^{-3}$ 
* **Loss Function**: Focal Loss ($\gamma$=2, $\alpha$=0.25)
* **Early Stopping**: Patience of 15 epochs

### Accuracy Comparison

The final test accuracies on the FER2013 test set were as follows:

| Model                     | Test Accuracy (%) |
| :------------------------ | :---------------- |
| Custom CNN                | 61.54             |
| MobileNetV2 (Fine-Tuned)  | 57.06             |
| MobileNetV2 (Head Only)   | 42.23             |

### Training Behavior

* **Custom CNN** reached stable convergence earlier than MobileNetV2 models.
* **MobileNetV2 (Fine-Tuned)** required more epochs to stabilize but achieved more balanced generalization.
* **MobileNetV2 (Head Only)** plateaued quickly, underfitting due to frozen base layers.

### Prediction Samples

Qualitative analysis using sample predictions showed that while the Custom CNN often predicted the correct emotion, the fine-tuned MobileNetV2 outperformed its original version in recognizing subtle emotions like "disgust".

#### Disgust Emotion Prediction Sample (Custom CNN)

<br>
<p align="center">
  <img src="images/disgust_custom_cnn.png" alt="Disgust Prediction - Custom CNN" width="600"/>
  <br>
  <em>Figure 8. Sample prediction with Custom CNN</em>
</p>
<br>

#### Disgust Emotion Prediction Sample (MobileNetV2 Head Only)

<br>
<p align="center">
  <img src="images/disgust_mobilenetv2_head_only.png" alt="Disgust Prediction - MobileNetV2 Head Only" width="600"/>
  <br>
  <em>Figure 9. Sample prediction with MobileNetV2 (Head Only)</em> 
</p>
<br>

#### Disgust Emotion Prediction Sample (MobileNetV2 Fine-Tuned)

<br>
<p align="center">
  <img src="images/disgust_mobilenetv2_fine_tuned.png" alt="Disgust Prediction - MobileNetV2 Fine-Tuned" width="600"/>
  <br>
  <em>Figure 10. Sample prediction with MobileNetV2 fine-tuned version</em>
</p>
<br>

#### Fear Emotion Prediction Sample (Custom CNN)

<br>
<p align="center">
  <img src="images/fear_custom_cnn.png" alt="Fear Prediction - Custom CNN" width="600"/>
  <br>
  <em>Figure 11. Sample prediction with Custom CNN</em>
</p>
<br>

#### Fear Emotion Prediction Sample (MobileNetV2 Head Only)

<br>
<p align="center">
  <img src="images/fear_mobilenetv2_head_only.png" alt="Fear Prediction - MobileNetV2 Head Only" width="600"/>
  <br>
  <em>Figure 12. Sample prediction with MobileNetV2 (Head Only)</em>
</p>
<br>

#### Fear Emotion Prediction Sample (MobileNetV2 Fine-Tuned)

<br>
<p align="center">
  <img src="images/fear_mobilenetv2_fine_tuned.png" alt="Fear Prediction - MobileNetV2 Fine-Tuned" width="600"/>
  <br>
  <em>Figure 13. Sample prediction with MobileNetV2 fine-tuned version</em>
</p>
<br>

These insights confirm that model depth and training strategy significantly impact FER performance, particularly under imbalanced data conditions. The fine-tuned MobileNetV2, being a more detailed and deep model, predicted correct emotions with high confidence and almost zero prediction confidence for false choices, as seen in the comparison of Figures 8 & 10, and Figures 11 & 13.

## Conclusions

This study developed and evaluated three CNN architectures for facial expression recognition on the FER2013 dataset: a Custom CNN, MobileNetV2 (Head Only), and a Fine-Tuned MobileNetV2. The Custom CNN achieved the best overall test accuracy at 61.54%, outperforming the fine-tuned and head-only MobileNetV2 models.

The use of Focal Loss effectively mitigated the impact of class imbalance, especially for underrepresented classes such as "disgust". Fine-tuning the MobileNetV2 backbone improved accuracy compared to freezing its base layers, highlighting the importance of end-to-end learning for facial expression tasks.

Despite competitive results, the models still fell short of state-of-the-art benchmarks (~73%). This gap is largely due to architectural simplicity, limited data diversity, and a lack of ensemble techniques. Additionally, confusion matrices revealed persistent misclassification between visually similar emotions like "fear" and "sad".

### Future Directions

* Explore advanced augmentation techniques and synthetic data generation to balance class distribution.
* Implement ensemble learning and attention mechanisms to enhance feature extraction.
* Experiment with transfer learning from larger, more diverse datasets such as AffectNet.
* Deploy lightweight models on edge devices for real-time FER applications.

These directions aim to bridge the performance gap and enhance the robustness of facial expression recognition systems under real-world constraints.

## References

* Khaireddin, M., & Chen, Y. (2021). Facial expression recognition using VGG-based CNN models. arXiv. [https://arxiv.org/pdf/2105.03588.pdf](https://arxiv.org/pdf/2105.03588.pdf) 
* WuJie1010. (2021). Facial-Expression-Recognition.Pytorch. GitHub. [https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch) 
* Gao, Z. (2024). Comparison of CNN and ResNet Neural Networks on the Performance of Facial Expression Recognition. *Highlights in Science, Engineering and Technology*, *94*. [https://drpress.org/ojs/index.php/HSET/article/view/20483/20042](https://drpress.org/ojs/index.php/HSET/article/view/20483/20042) 
* Santoso, B. E., & Kusuma, G. P. (2022). Facial Emotion Recognition on FER2013 Using VGGSpinalNet. *Journal of Theoretical and Applied Information Technology*, *100*(7), 2193–2199. [http://www.jatit.org/volumes/Vol100No7/10Vol100No7.pdf](http://www.jatit.org/volumes/Vol100No7/10Vol100No7.pdf)
* Yan, C., Zhang, X., & Wang, Q. (n.d.). Face Expression Recognition Based on Improved Mobilenext. SSRN. Retrieved from [http://dx.doi.org/10.2139/ssrn.4220699](http://dx.doi.org/10.2139/ssrn.4220699)
* Bhagat, D., Vakil, A., Gupta, R. K., & Kumar, A. (2024). Facial Emotion Recognition (FER) using Convolutional Neural Network (CNN). *Procedia Computer Science*, *235*, 2079–2089. [https://doi.org/10.1016/j.procs.2024.04.197](https://doi.org/10.1016/j.procs.2024.04.197)
