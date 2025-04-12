# 🃏 Playing Card Image Classifier (Transfer Learning + TensorFlow)

A deep learning model that classifies **53 playing card types** using **MobileNetV2** and TensorFlow.

## ✅ Features

- Applied data augmentation and caching for efficiency
- Analyzed class imbalance and built a balanced pipeline
- Trained using learning rate decay and early stopping
- Visualized predictions and model performance over epochs

## 🗂️ Dataset Structure

- `train/`
- `valid/`
- `test/`
- `cards.csv`

## 📊 Visuals

- 📈 Accuracy and loss trends
- 🎯 Sample predictions vs. ground truth
- 🌀 Augmented image previews

## 🧠 Model Architecture

```python
MobileNetV2
→ GlobalAveragePooling2D
→ Dense(512, relu)
→ Dropout(0.5)
→ Dense(53, softmax)
