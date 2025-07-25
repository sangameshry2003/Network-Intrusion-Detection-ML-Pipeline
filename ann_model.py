# Artificial Neural Network Model for Network Intrusion Detection
# Notebook: 04_ann_model.ipynb

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("🧠 Artificial Neural Network Model Training for Network Intrusion Detection")
print("=" * 80)

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

# Initialize ANN model
print(f"\n🧠 Initializing Artificial Neural Network...")
ann_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

# 5-fold cross-validation
print(f"\n🔄 Performing 5-fold cross-validation...")
cv_scores = cross_val_score(
    ann_model, X_train, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

print(f"CV Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train the model
print(f"\n🏋️ Training ANN model on full training set...")
ann_model.fit(X_train, y_train)
print("✅ Training completed!")

# Training information
print(f"\n🧠 ANN Training Information:")
print(f"Number of iterations: {ann_model.n_iter_}")
print(f"Number of layers: {ann_model.n_layers_}")
print(f"Number of outputs: {ann_model.n_outputs_}")
print(f"Loss: {ann_model.loss_:.6f}")

# Make predictions
print(f"\n🔮 Making predictions on test set...")
y_pred = ann_model.predict(X_test)
y_pred_proba = ann_model.predict_proba(X_test)

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

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.title('ANN - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/confusion_ann.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance approximation using permutation importance
print(f"\n🏆 Calculating feature importance (permutation-based)...")
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    ann_model, X_test, y_test, 
    n_repeats=10, 
    random_state=42
)

feature_importance = pd.DataFrame({
    'feature': feature_list,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualize feature importances
plt.figure(figsize=(10, 6))
feature_importance_plot = feature_importance.head(10)
plt.barh(range(len(feature_importance_plot)), feature_importance_plot['importance'])
plt.yticks(range(len(feature_importance_plot)), feature_importance_plot['feature'])
plt.xlabel('Permutation Importance')
plt.title('ANN - Top 10 Feature Importances (Permutation-based)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/feature_importance_ann.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot training loss curve if available
if hasattr(ann_model, 'loss_curve_'):
    plt.figure(figsize=(10, 6))
    plt.plot(ann_model.loss_curve_)
    plt.title('ANN Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/training_loss_ann.png', dpi=300, bbox_inches='tight')
    plt.show()

# Save the model
print(f"\n💾 Saving ANN model...")
joblib.dump(ann_model, 'artifacts/model_ann.pkl')

# Save evaluation results
results = {
    'model_name': 'ANN',
    'accuracy': accuracy,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'cv_scores': cv_scores.tolist(),
    'classification_report': class_report,
    'confusion_matrix': cm.tolist(),
    'feature_importance': feature_importance.to_dict('records'),
    'hidden_layer_sizes': list(ann_model.hidden_layer_sizes),
    'activation': ann_model.activation,
    'solver': ann_model.solver,
    'n_iter': int(ann_model.n_iter_),
    'n_layers': int(ann_model.n_layers_),
    'loss': float(ann_model.loss_)
}

with open('artifacts/results_ann.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Saved successfully:")
print("   📁 artifacts/model_ann.pkl")
print("   📁 artifacts/results_ann.json")
print("   📁 plots/confusion_ann.png")
print("   📁 plots/feature_importance_ann.png")
if hasattr(ann_model, 'loss_curve_'):
    print("   📁 plots/training_loss_ann.png")

print(f"\n🎉 ANN model training completed!")
print(f"Final Test Accuracy: {accuracy:.4f}")
print("=" * 80)