# ECG Heartbeat Classification using 1D-CNN and Tuned Focal Loss

An end-to-end PyTorch implementation of a 1D Convolutional Neural Network (1D-CNN) designed to classify electrocardiogram (ECG) heartbeats using the MIT-BIH Arrhythmia Database. This repository addresses the severe class imbalance inherent in medical data by utilizing a customized, balanced Focal Loss function.

## 📌 Project Overview
ECG classification is crucial for detecting cardiac arrhythmias. However, dataset imbalance often causes traditional deep learning models to favor the majority class (Normal beats) while misclassifying critical minority classes (such as Fusion or SVPB beats). 

This project implements:
- **1D-CNN Architecture**: Optimized for sequential extraction of time-series ECG signal topologies.
- **Custom Tuned Focal Loss**: Dynamically downweights easy-to-classify normal samples to focus training attention on hard, rare cardiac anomalies.
- **Dynamic Class Weighting**: Integrated class parameters to balance Precision and Recall tradeoffs effectively.

---

## 📊 Dataset Structure
The architecture evaluates inputs formatted from the **MIT-BIH Heartbeat Categorization Dataset**, structured into 187 sequence lengths mapping to 5 distinct labels:
- **Class 0**: Normal Beat
- **Class 1**: Supraventricular Premature Beat (SVPB)
- **Class 2**: Premature Ventricular Contraction (PVC)
- **Class 3**: Fusion Beat
- **Class 4**: Unknown / Unclassified Beat

---

## 🚀 Key Performance Results
The network achieves an overall baseline **Test Accuracy of 97.54%** within just 5 training epochs, showcasing excellent minority class capture:
- **Normal (Class 0)**: ~99.26% Precision | ~98.14% Recall
- **SVPB (Class 1)**: ~84.27% Recall (Mitigated False Negatives)
- **Fusion (Class 3)**: ~85.16% Recall 

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python installed along with the required libraries. Install dependencies via pip:
```bash
pip install torch pandas numpy scikit-learn seaborn matplotlib
```

### 2. Dataset Placement
Download the `mitbih_train.csv` file and place it directly into your project root directory or upload it to your Google Colab environment.

### 3. Execution
Run the pipeline script to execute data preprocessing, model compilation, training, and metrics visualization:
```bash
python ecg_classification.py
```

---

## 🧬 Model Architecture
```text
Input Signal [Batch, 1, 187]
   │
   ├──► Conv1d (32 filters, kernel=5) ──► ReLU ──► MaxPool1d (kernel=2)
   │
   ├──► Conv1d (64 filters, kernel=5) ──► ReLU ──► MaxPool1d (kernel=2)
   │
   ├──► Flatten [Batch, 64 * 46]
   │
   ├──► Linear (128 units) ──► ReLU
   │
   └──► Linear Output (5 Target Classes)
```

---

## 📈 Visualizations
Upon completing evaluation runs, the pipeline generates a graphical **Confusion Matrix** using Seaborn (`cmap='Greens'`) alongside a comprehensive `classification_report` summarizing precision, recall, and F1-scores for all 5 arrhythmia types.
