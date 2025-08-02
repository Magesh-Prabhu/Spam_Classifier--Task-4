# 📧 Spam Email Detection using Scikit-learn

This is a beginner-friendly machine learning project in Python that detects whether a message is **spam** or **not spam (ham)** using the **Logistic Regression** model and **Scikit-learn**.

## 🚀 Features

- Load and clean the SMS Spam dataset
- Text preprocessing with `TfidfVectorizer`
- Train/test split and model training
- Performance evaluation (Accuracy, Confusion Matrix, Classification Report)

## 📁 Dataset

The dataset used is the public [SMS Spam Collection](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv) with two columns:
- `label`: spam or ham
- `message`: the actual text message

## 🛠 Requirements

Install Python dependencies:

```bash
pip install pandas scikit-learn

