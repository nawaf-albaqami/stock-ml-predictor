import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

print("Downloading stock data...")

# إنشاء مجلد model إذا غير موجود
os.makedirs("model", exist_ok=True)

# تحميل البيانات
data = yf.download("AAPL", start="2022-01-01")

# =========================
# Feature Engineering
# =========================

# Moving Average
data["MA_10"] = data["Close"].rolling(window=10).mean()

# RSI
delta = data["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

# الهدف: هل السعر سيرتفع غداً؟
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

# حذف القيم الفارغة
data = data.dropna()

# =========================
# اختيار الميزات
# =========================

features = ["Open", "High", "Low", "Close", "Volume", "MA_10", "RSI"]

X = data[features]
y = data["Target"]

# تقسيم زمني (بدون shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("Training model...")

# =========================
# تدريب النموذج
# =========================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# التقييم
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(cm)

# Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross Validation Accuracy: {cv_scores.mean():.2f}")

# =========================
# حفظ النموذج
# =========================

joblib.dump(model, "model/stock_model.pkl")

print("\nModel saved successfully 🚀")