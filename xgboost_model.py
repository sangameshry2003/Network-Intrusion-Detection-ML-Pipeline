# XGBoost Model for Network Intrusion Detection
# Notebook: 01_xgboost_model.ipynb

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("🚀 XGBoost Model Training for Network Intrusion Detection")
print("=" * 65)

# Load preprocessed data
print("📂 Loading preprocessed data from artifacts/...")
X_train = joblib.load('artifacts/X_train.pkl')
X_test = joblib.load('artifacts/X_test.pkl')
y_train = joblib.load('artifacts/y_train.pkl')
y_test = joblib.load('artifacts/y_test.pkl')
feature_list = joblib.load('artifacts/feature_list.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

print(f"✅ Data loaded successfully!")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Features: {feature_list}")

# Initialize XGBoost model
print(f"\n🌳 Initializing XGBoost Classifier...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

# 5-fold cross-validation
print(f"\n🔄 Performing 5-fold cross-validation...")
cv_scores = cross_val_score(
    xgb_model, X_train, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

print(f"CV Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train the model
print(f"\n🏋️ Training XGBoost model on full training set...")
xgb_model.fit(X_train, y_train)
print("✅ Training completed!")

# Make predictions
print(f"\n🔮 Making predictions on test set...")
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification report
print(f"\n📊 Classification Report:")
class_report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Confusion matrix
print(f"\n🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importances
print(f"\n🏆 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': feature_list,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.title('XGBoost - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/confusion_xgboost.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize feature importances
plt.figure(figsize=(10, 6))
feature_importance_plot = feature_importance.head(10)
plt.barh(range(len(feature_importance_plot)), feature_importance_plot['importance'])
plt.yticks(range(len(feature_importance_plot)), feature_importance_plot['feature'])
plt.xlabel('Feature Importance')
plt.title('XGBoost - Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/feature_importance_xgboost.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the model
print(f"\n💾 Saving XGBoost model...")
joblib.dump(xgb_model, 'artifacts/model_xgboost.pkl')

# Save evaluation results
results = {
    'model_name': 'XGBoost',
    'accuracy': accuracy,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'cv_scores': cv_scores.tolist(),
    'classification_report': class_report,
    'confusion_matrix': cm.tolist(),
    'feature_importance': feature_importance.to_dict('records')
}

with open('artifacts/results_xgboost.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Saved successfully:")
print("   📁 artifacts/model_xgboost.pkl")
print("   📁 artifacts/results_xgboost.json")
print("   📁 plots/confusion_xgboost.png")
print("   📁 plots/feature_importance_xgboost.png")

print(f"\n🎉 XGBoost model training completed!")
print(f"Final Test Accuracy: {accuracy:.4f}")
print("=" * 65)