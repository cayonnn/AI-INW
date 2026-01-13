"""
train_xgboost.py
================
XGBoost Imitation Learning Trainer

Mental Model:
- Rule-Based SignalEngine v2 = Teacher / Oracle
- XGBoost = Student  
- Risk Guard = Principal (final authority)

Usage:
    python src/ai/train_xgboost.py
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
DATA_PATH = "data/imitation_dataset.csv"
MODEL_PATH = "models/xgb_signal.model"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES = 20  # Minimum samples required

# =========================
# FEATURE COLUMNS
# =========================
FEATURE_COLUMNS = [
    "ema_fast",
    "ema_slow",
    "ema_spread",
    "ema_slope",
    "atr",
    "atr_ratio",
    "atr_threshold",
    "htf_trend",
    "volatility_ok",
    "hour",
    "day_of_week",
]

# =========================
# LOAD DATA
# =========================
print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"   Total samples: {len(df)}")
print(f"   BUY:  {len(df[df['action'] == 'BUY'])}")
print(f"   SELL: {len(df[df['action'] == 'SELL'])}")
print(f"   HOLD: {len(df[df['action'] == 'HOLD'])}")

if len(df) < MIN_SAMPLES:
    print(f"\n❌ Not enough samples. Need at least {MIN_SAMPLES}, got {len(df)}")
    print("   Run sandbox mode to collect more data:")
    print("   python main.py --sandbox --interval 30")
    exit(1)

# Map signal to class
signal_map = {
    "SELL": 0,
    "HOLD": 1,
    "BUY": 2
}
df["signal_class"] = df["action"].map(signal_map)

# Check for balanced classes
unique_classes = df["signal_class"].nunique()
if unique_classes < 2:
    print(f"\n⚠️ Warning: Only {unique_classes} class(es) in dataset.")
    print("   Need at least 2 classes for proper training.")
    print("   Continue collecting diverse signals...")

# =========================
# FEATURES / LABELS
# =========================
print("\n🔧 Preparing features...")

# Select only available feature columns
available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
X = df[available_features].fillna(0)
y = df["signal_class"]

print(f"   Features: {len(available_features)}")
print(f"   {available_features}")

# =========================
# TRAIN / TEST SPLIT
# =========================
print("\n📊 Splitting data...")

# Handle stratify with single class
if df["signal_class"].nunique() > 1:
    stratify = y
else:
    stratify = None
    print("   ⚠️ Single class - cannot stratify")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify
)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# =========================
# XGBOOST MODEL
# =========================
print("\n🚀 Training XGBoost...")

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    random_state=RANDOM_STATE,
    verbosity=0,
)

# =========================
# TRAIN
# =========================
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# =========================
# EVALUATION
# =========================
print("\n📈 Evaluating...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Accuracy
train_acc = (model.predict(X_train) == y_train).mean()
test_acc = (y_pred == y_test).mean()

print(f"   Train Accuracy: {train_acc:.1%}")
print(f"   Test Accuracy:  {test_acc:.1%}")

# Classification Report
print("\n📋 Classification Report:")
try:
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = ["SELL", "HOLD", "BUY"]
    actual_labels = [target_names[i] for i in unique_labels if i < len(target_names)]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=actual_labels))
except Exception as e:
    print(f"   (Report skipped: {e})")

# Confusion Matrix
print("📉 Confusion Matrix:")
try:
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
except:
    print("   (Matrix skipped)")

# Feature Importance
print("\n🔥 Top Features:")
feature_imp = sorted(zip(available_features, model.feature_importances_), 
                     key=lambda x: x[1], reverse=True)
for feat, imp in feature_imp[:5]:
    print(f"   {feat}: {imp:.3f}")

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)
model.save_model(MODEL_PATH)

print(f"\n✅ Model saved to: {MODEL_PATH}")

# =========================
# QUALITY CHECK
# =========================
print("\n" + "=" * 50)
if test_acc > 0.95:
    print("⚠️ CAUTION: Accuracy > 95% may indicate overfitting!")
    print("   Recommend: Collect more diverse data")
elif test_acc > 0.80:
    print("✅ GOOD: Model accuracy is in healthy range")
elif test_acc > 0.60:
    print("📊 FAIR: Model needs more training data")
else:
    print("❌ POOR: Model needs significantly more data")
print("=" * 50)
