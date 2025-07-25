# Naive Bayes Model for Network Intrusion Detection
# Notebook: 07_naive_bayes_model.ipynb

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print("🎯 Naive Bayes Model Training for Network Intrusion Detection")
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

# Initialize Naive Bayes model
print(f"\n🎯 Initializing Gaussian Naive Bayes Classifier...")
nb_model = GaussianNB(
    var_smoothing=1e-9  # Default smoothing parameter
)

# 5-fold cross-validation
print(f"\n🔄 Performing 5-fold cross-validation...")
cv_scores = cross_val_score(
    nb_model, X_train, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

print(f"CV Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train the model
print(f"\n🏋️ Training Naive Bayes model on full training set...")
nb_model.fit(X_train, y_train)
print("✅ Training completed!")

# Make predictions
print(f"\n🔮 Making predictions on test set...")
y_pred = nb_model.predict(X_test)
y_pred_proba = nb_model.predict_proba(X_test)

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

# Naive Bayes specific information
print(f"\n🎯 Naive Bayes Model Information:")
print(f"Number of classes: {nb_model.classes_.shape[0]}")
print(f"Classes: {nb_model.classes_}")
print(f"Var smoothing: {nb_model.var_smoothing}")

# Feature statistics for each class
print(f"\n📊 Feature Statistics by Class:")
class_stats = pd.DataFrame({
    'Feature': feature_list,
    'Benign_Mean': nb_model.theta_[0],
    'Malicious_Mean': nb_model.theta_[1],
    'Benign_Var': nb_model.var_[0],
    'Malicious_Var': nb_model.var_[1]
})
print(class_stats.round(4))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.title('Naive Bayes - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/confusion_naive_bayes.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance using permutation importance
print(f"\n🏆 Calculating feature importance (permutation-based)...")
perm_importance = permutation_importance(
    nb_model, X_test, y_test, 
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
plt.title('Naive Bayes - Top 10 Feature Importances (Permutation-based)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/feature_importance_naive_bayes.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize class means comparison
plt.figure(figsize=(12, 8))
x_pos = np.arange(len(feature_list))
width = 0.35

plt.bar(x_pos - width/2, class_stats['Benign_Mean'], width, 
        label='Benign', alpha=0.8, color='blue')
plt.bar(x_pos + width/2, class_stats['Malicious_Mean'], width, 
        label='Malicious', alpha=0.8, color='red')

plt.xlabel('Features')
plt.ylabel('Mean Values')
plt.title('Naive Bayes - Feature Means by Class')
plt.xticks(x_pos, feature_list, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('plots/class_means_naive_bayes.png', dpi=300, bbox_inches='tight')
plt.show()

# Probability calibration analysis
print(f"\n📈 Prediction Probability Analysis:")
prob_stats = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Prob_Benign': y_pred_proba[:, 0],
    'Prob_Malicious': y_pred_proba[:, 1]
})

print("Probability distribution by true class:")
print(prob_stats.groupby('True_Label')[['Prob_Benign', 'Prob_Malicious']].describe())

# Plot probability distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Benign class probabilities
benign_samples = prob_stats[prob_stats['True_Label'] == 0]
axes[0].hist(benign_samples['Prob_Benign'], bins=20, alpha=0.7, label='Correct', color='blue')
axes[0].hist(benign_samples['Prob_Malicious'], bins=20, alpha=0.7, label='Incorrect', color='red')
axes[0].set_title('Probability Distribution - True Benign Samples')
axes[0].set_xlabel('Probability')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Malicious class probabilities
malicious_samples = prob_stats[prob_stats['True_Label'] == 1]
if len(malicious_samples) > 0:
    axes[1].hist(malicious_samples['Prob_Malicious'], bins=20, alpha=0.7, label='Correct', color='red')
    axes[1].hist(malicious_samples['Prob_Benign'], bins=20, alpha=0.7, label='Incorrect', color='blue')
    axes[1].set_title('Probability Distribution - True Malicious Samples')
    axes[1].set_xlabel('Probability')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, 'No malicious samples\nin test set', 
                ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Probability Distribution - True Malicious Samples')

plt.tight_layout()
plt.savefig('plots/probability_distribution_naive_bayes.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the model
print(f"\n💾 Saving Naive Bayes model...")
joblib.dump(nb_model, 'artifacts/model_naive_bayes.pkl')

# Save evaluation results
results = {
    'model_name': 'Naive_Bayes',
    'accuracy': float(accuracy),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'cv_scores': cv_scores.tolist(),
    'classification_report': class_report,
    'confusion_matrix': cm.tolist(),
    'feature_importance': feature_importance.to_dict('records'),
    'n_classes': int(nb_model.classes_.shape[0]),
    'classes': nb_model.classes_.tolist(),
    'var_smoothing': float(nb_model.var_smoothing),
    'class_statistics': class_stats.to_dict('records')
}

with open('artifacts/results_naive_bayes.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Saved successfully:")
print("   📁 artifacts/model_naive_bayes.pkl")
print("   📁 artifacts/results_naive_bayes.json")
print("   📁 plots/confusion_naive_bayes.png")
print("   📁 plots/feature_importance_naive_bayes.png")
print("   📁 plots/class_means_naive_bayes.png")
print("   📁 plots/probability_distribution_naive_bayes.png")

print(f"\n🎉 Naive Bayes model training completed!")
print(f"Final Test Accuracy: {accuracy:.4f}")
print("=" * 65)