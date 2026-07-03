# High-Dimensional Text Classification using TF-IDF Features and Deep Neural Networks 

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Supported-orange.svg)

> **Research Paper**: This repository contains the code for our research paper published via the IEEE conference held at Chandigarh University 2026.
> [Read the full paper here](https://ieeexplore.ieee.org/document/11496177).

##  Project Overview

This project implements a robust supervised text classification pipeline for categorizing news documents into one of 20 distinct topic categories. Utilizing the **20 Newsgroups dataset**, the model transforms raw text documents into high-dimensional TF-IDF (Term Frequency-Inverse Document Frequency) vectors, which are then passed into a dense Keras-based Multilayer Perceptron (MLP) for accurate classification.

##  Key Features

- **End-to-End Pipeline**: From raw text preprocessing to deep learning model evaluation.
- **High-Dimensional Feature Extraction**: Leverages `TfidfVectorizer` for unigrams and bigrams, creating up to 50,000 features.
- **Deep Neural Network Classifier**: Custom MLP architecture built with TensorFlow/Keras.
- **Comprehensive Evaluation**: Detailed metrics including precision, recall, F1-score, and per-class accuracy.

##  Dataset

The model is trained on the built-in `fetch_20newsgroups` dataset from `scikit-learn`.

- **Classes**: 20 distinct news categories.
- **Split**: 80% Training / 20% Testing (Stratified).
- **Format**: Raw text mapped to integer class labels.

##  Methodology & Architecture

### Text Vectorization
The text is vectorized using scikit-learn's `TfidfVectorizer` with the following configuration:
- English stop-word removal.
- N-gram range: (1, 2) (Unigrams and Bigrams).
- Max Document Frequency: 0.7
- Maximum Features: 50,000

### Model Architecture
The classifier is a Sequential MLP implemented in Keras:

| Layer | Output Size | Activation | Notes |
| :--- | :--- | :--- | :--- |
| **Input** | 50,000 | None | TF-IDF feature vector |
| **Dense** | 512 | ReLU | Hidden layer 1 |
| **Dropout** | 512 | None | Rate: 0.3 |
| **Dense** | 256 | ReLU | Hidden layer 2 |
| **Dropout** | 256 | None | Rate: 0.3 |
| **Output** | 20 | Softmax | Probabilities across 20 classes |

*Total Parameters: ~25.7 Million*

##  Results & Visualizations

The repository includes several visual artifacts demonstrating the model's performance:
- `accuracy_loss.png` - Training/Validation metrics across epochs.
- `confusion_matrix.png` - Visualizing class-wise predictions.
- `per_class_accuracy.png` - Granular view of accuracy across all 20 categories.
- `text_classification_architecture.png` & `categorization_system.png` - Architectural diagrams.

##  Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed. Install the required dependencies:

```bash
pip install numpy scikit-learn tensorflow jupyter matplotlib
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/nandeeshib/High-Dimensional-Text-Classification-using-TF-IDF-Features-and-Deep-Neural-Network.git
   cd High-Dimensional-Text-Classification-using-TF-IDF-Features-and-Deep-Neural-Network
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `TF-IDF_Model.ipynb` and run all cells.

##  Notes

- Converting a 50,000-feature sparse TF-IDF matrix into a dense array can be highly memory-intensive.
- The dataset is downloaded automatically on the first run, requiring internet access.
- Results may vary slightly depending on your hardware and software versions.
