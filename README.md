# 📈 Stock ML Predictor

End-to-end Machine Learning pipeline for stock movement prediction using Random Forest and technical indicators.

## 🚀 Project Overview
This project builds a machine learning model to predict stock price direction (Up/Down) using:

- Historical stock data via yfinance
- Technical indicators (SMA, RSI, etc.)
- Random Forest Classifier
- Cross-validation
- Model persistence using joblib

## 🧠 ML Pipeline
1. Data collection
2. Feature engineering
3. Train/Test split
4. Model training
5. Evaluation (Accuracy + Confusion Matrix)
6. Cross-validation
7. Model saving

## 📊 Example Output
- Model Accuracy: ~0.49
- Cross Validation Accuracy: ~0.48

## 🛠 Tech Stack
- Python 3.12
- pandas
- numpy
- scikit-learn
- yfinance
- joblib

## 📦 Installation

```bash

pip install -r requirements.txt

## 🔮 Future Improvements
- Hyperparameter tuning
- Feature importance visualization
- Backtesting trading strategy
- Deploy as a Streamlit web app

