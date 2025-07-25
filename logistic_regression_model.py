# Logistic Regression Model for Network Intrusion Detection
# Notebook: 08_logistic_regression_model.ipynb

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("📈 Logistic Regression Model Training for Network Intrusion Detection")
print("=" * 75)

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

# Initialize Logistic Regression model
print(f"\n📈 Initializing Logistic Regression Classifier...")
lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)

# 5-fold cross-validation
print(f"\n🔄 Performing 5-fold cross-validation...")
cv_scores = cross_val_score(
    lr_model, X_train, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

print(f"CV Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train the model
print(f"\n🏋️ Training Logistic Regression model on full training set...")
lr_model.fit(X_train, y_train)
print("✅ Training completed!")

# Make predictions
print(f"\n🔮 Making predictions on test set...")
y_pred = lr_model.predict(X_test)
y_pred_proba = lr_model.predict_proba(X_test)

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

# Logistic Regression specific information
print(f"\n📈 Logistic Regression Model Information:")
print(f"Regularization parameter C: {lr_model.C}")
print(f"Penalty: {lr_model.penalty}")
print(f"Solver: {lr_model.solver}")
print(f"Number of iterations: {lr_model.n_iter_[0]}")
print(f"Intercept: {lr_model.intercept_[0]:.4f}")

# Feature importances using coefficients
print(f"\n🏆 Feature Importances (Coefficient-based):")
feature_importance = pd.DataFrame({
    'feature': feature_list,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(feature_importance)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', 
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.title('Logistic Regression - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/confusion_logistic_regression.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize feature coefficients
plt.figure(figsize=(12, 8))
feature_coef_plot = feature_importance.head(10)
colors = ['red' if x < 0 else 'blue' for x in feature_coef_plot['coefficient']]
plt.barh(range(len(feature_coef_plot)), feature_coef_plot['coefficient'], color=colors)
plt.yticks(range(len(feature_coef_plot)), feature_coef_plot['feature'])
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression - Feature Coefficients (Top 10 by Magnitude)')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/feature_coefficients_logistic_regression.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize feature importance by absolute value
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_coef_plot)), feature_coef_plot['abs_coefficient'])
plt.yticks(range(len(feature_coef_plot)), feature_coef_plot['feature'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Logistic Regression - Feature Importance (Absolute Coefficients)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/feature_importance_logistic_regression.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve Analysis
print(f"\n📈 ROC Curve Analysis:")
if len(np.unique(y_test)) > 1:  # Only if we have both classes
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curve_logistic_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")
else:
    print("ROC curve cannot be computed - only one class in test set")
    roc_auc = None

# Probability calibration analysis
print(f"\n📊 Prediction Probability Analysis:")
prob_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Prob_Benign': y_pred_proba[:, 0],
    'Prob_Malicious': y_pred_proba[:, 1]
})

print("Probability statistics:")
print(prob_df[['Prob_Benign', 'Prob_Malicious']].describe())

# Plot probability distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(prob_df['Prob_Benign'], bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title('Predicted Probability for Benign Class')
plt.xlabel('Probability')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(prob_df['Prob_Malicious'], bins=20, alpha=0.7, color='red', edgecolor='black')
plt.title('Predicted Probability for Malicious Class')
plt.xlabel('Probability')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('plots/probability_histogram_logistic_regression.png', dpi=300, bbox_inches='tight')
plt.show()

# Decision boundary analysis
print(f"\n🎯 Decision Boundary Analysis:")
decision_scores = lr_model.decision_function(X_test)
print(f"Decision scores - Mean: {decision_scores.mean():.4f}, Std: {decision_scores.std():.4f}")
print(f"Decision scores - Min: {decision_scores.min():.4f}, Max: {decision_scores.max():.4f}")

plt.figure(figsize=(10, 6))
plt.hist(decision_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
plt.title('Logistic Regression - Decision Scores Distribution')
plt.xlabel('Decision Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/decision_scores_logistic_regression.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature correlation with target
print(f"\n🔍 Feature Correlation Analysis:")
correlation_df = pd.DataFrame({
    'Feature': feature_list,
    'Coefficient': lr_model.coef_[0],
    'Interpretation': ['Increases malicious probability' if x > 0 else 'Decreases malicious probability' 
                      for x in lr_model.coef_[0]]
})

print(correlation_df.sort_values('Coefficient', key=abs, ascending=False))

# Save the model
print(f"\n💾 Saving Logistic Regression model...")
joblib.dump(lr_model, 'artifacts/model_logistic_regression.pkl')

# Save evaluation results
results = {
    'model_name': 'Logistic_Regression',
    'accuracy': float(accuracy),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'cv_scores': cv_scores.tolist(),
    'classification_report': class_report,
    'confusion_matrix': cm.tolist(),
    'feature_importance': feature_importance.to_dict('records'),
    'C': float(lr_model.C),
    'penalty': lr_model.penalty,
    'solver': lr_model.solver,
    'n_iter': int(lr_model.n_iter_[0]),
    'intercept': float(lr_model.intercept_[0]),
    'roc_auc': float(roc_auc) if roc_auc is not None else None,
    'coefficients': lr_model.coef_[0].tolist()
}

with open('artifacts/results_logistic_regression.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Saved successfully:")
print("   📁 artifacts/model_logistic_regression.pkl")
print("   📁 artifacts/results_logistic_regression.json")
print("   📁 plots/confusion_logistic_regression.png")
print("   📁 plots/feature_coefficients_logistic_regression.png")
print("   📁 plots/feature_importance_logistic_regression.png")
if roc_auc is not None:
    print("   📁 plots/roc_curve_logistic_regression.png")
print("   📁 plots/probability_histogram_logistic_regression.png")
print("   📁 plots/decision_scores_logistic_regression.png")

print(f"\n🎉 Logistic Regression model training completed!")
print(f"Final Test Accuracy: {accuracy:.4f}")
if roc_auc is not None:
    print(f"ROC AUC: {roc_auc:.4f}")
print("=" * 75)