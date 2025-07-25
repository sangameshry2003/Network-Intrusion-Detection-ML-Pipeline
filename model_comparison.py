# Model Comparison and Summary for Network Intrusion Detection
# Notebook: 05_model_comparison_summary.ipynb

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("📊 Model Comparison and Summary for Network Intrusion Detection")
print("=" * 75)

# Define models to compare
models = ['xgboost', 'random_forest', 'svm', 'ann', 'decision_tree', 'naive_bayes', 'logistic_regression', 'knn']
model_names = ['XGBoost', 'Random Forest', 'SVM', 'ANN', 'Decision Tree', 'Naive Bayes', 'Logistic Regression', 'KNN']

# Load all results
print("📂 Loading model results...")
all_results = {}
for model in models:
    try:
        with open(f'artifacts/results_{model}.json', 'r') as f:
            all_results[model] = json.load(f)
        print(f"✅ Loaded {model} results")
    except FileNotFoundError:
        print(f"❌ Results for {model} not found")

print(f"\nLoaded results for {len(all_results)} models")

# Create comparative performance table
print("\n📋 Creating comparative performance table...")
comparison_data = []

for model, results in all_results.items():
    model_name = results['model_name']
    accuracy = results['accuracy']
    cv_mean = results['cv_mean']
    cv_std = results['cv_std']
    
    # Extract precision, recall, f1 for class 1 (malicious)
    class_report = results['classification_report']
    if '1' in class_report:  # Malicious class
        precision = class_report['1']['precision']
        recall = class_report['1']['recall']
        f1_score = class_report['1']['f1-score']
    else:
        precision = recall = f1_score = 0
    
    comparison_data.append({
        'Model': model_name,
        'Test_Accuracy': accuracy,
        'CV_Mean': cv_mean,
        'CV_Std': cv_std,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)

print("\n🏆 Model Performance Comparison:")
print("=" * 80)
print(comparison_df.round(4).to_string(index=False))

# Find best performing model
best_model = comparison_df.iloc[0]
print(f"\n🥇 Best Performing Model: {best_model['Model']}")
print(f"   Test Accuracy: {best_model['Test_Accuracy']:.4f}")
print(f"   CV Score: {best_model['CV_Mean']:.4f} ± {best_model['CV_Std']:.4f}")
print(f"   Precision: {best_model['Precision']:.4f}")
print(f"   Recall: {best_model['Recall']:.4f}")
print(f"   F1-Score: {best_model['F1_Score']:.4f}")

# Create performance visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Accuracy comparison
axes[0, 0].bar(comparison_df['Model'], comparison_df['Test_Accuracy'], color='skyblue')
axes[0, 0].set_title('Test Accuracy')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].tick_params(axis='x', rotation=45)

# CV Score comparison
axes[0, 1].bar(comparison_df['Model'], comparison_df['CV_Mean'], 
               yerr=comparison_df['CV_Std'], color='lightgreen', capsize=5)
axes[0, 1].set_title('Cross-Validation Score')
axes[0, 1].set_ylabel('CV Score')
axes[0, 1].tick_params(axis='x', rotation=45)

# Precision-Recall comparison
axes[1, 0].bar(comparison_df['Model'], comparison_df['Precision'], 
               alpha=0.7, label='Precision', color='orange')
axes[1, 0].bar(comparison_df['Model'], comparison_df['Recall'], 
               alpha=0.7, label='Recall', color='red')
axes[1, 0].set_title('Precision vs Recall')
axes[1, 0].set_ylabel('Score')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend()

# F1-Score comparison
axes[1, 1].bar(comparison_df['Model'], comparison_df['F1_Score'], color='purple')
axes[1, 1].set_title('F1-Score')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Extract top features from tree-based models
print("\n🌳 Top Features from Tree-based Models:")
print("=" * 50)

tree_models = ['xgboost', 'random_forest', 'decision_tree']
for model in tree_models:
    if model in all_results:
        print(f"\n{all_results[model]['model_name']} - Top 3 Features:")
        features = all_results[model]['feature_importance'][:3]
        for i, feature in enumerate(features, 1):
            print(f"  {i}. {feature['feature']}: {feature['importance']:.4f}")

# Display confusion matrices
print(f"\n🎯 Confusion Matrices:")
print("=" * 30)

# Calculate grid size based on number of models
n_models = len([m for m in models if m in all_results])
cols = 4
rows = (n_models + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')

if rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

colors = ['Blues', 'Greens', 'Purples', 'Oranges', 'YlOrRd', 'Reds', 'plasma', 'viridis']

plot_idx = 0
for i, model in enumerate(models):
    if model in all_results:
        cm = np.array(all_results[model]['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=colors[i % len(colors)], 
                   xticklabels=['Benign', 'Malicious'],
                   yticklabels=['Benign', 'Malicious'],
                   ax=axes[plot_idx])
        axes[plot_idx].set_title(f'{all_results[model]["model_name"]}')
        axes[plot_idx].set_ylabel('True Label')
        axes[plot_idx].set_xlabel('Predicted Label')
        plot_idx += 1

# Hide empty subplots
for j in range(plot_idx, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('plots/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# Model-specific insights
print(f"\n🔍 Model-Specific Insights:")
print("=" * 40)

for model in models:
    if model in all_results:
        results = all_results[model]
        model_name = results['model_name']
        print(f"\n{model_name}:")
        
        if model == 'xgboost':
            print(f"  • Gradient boosting with 100 trees")
            print(f"  • Good balance of speed and accuracy")
            
        elif model == 'random_forest':
            n_trees = results.get('n_estimators', 100)
            print(f"  • Ensemble of {n_trees} decision trees")
            print(f"  • Robust against overfitting")
            
        elif model == 'svm':
            kernel = results.get('kernel', 'rbf')
            sv_ratio = results.get('support_vector_ratio', 0)
            print(f"  • Kernel: {kernel}")
            print(f"  • Support vector ratio: {sv_ratio:.4f}")
            
        elif model == 'ann':
            hidden_layers = results.get('hidden_layer_sizes', [])
            n_iter = results.get('n_iter', 0)
            print(f"  • Architecture: {hidden_layers}")
            print(f"  • Training iterations: {n_iter}")
            
        elif model == 'decision_tree':
            tree_depth = results.get('tree_depth', 0)
            n_leaves = results.get('n_leaves', 0)
            print(f"  • Tree depth: {tree_depth}")
            print(f"  • Number of leaves: {n_leaves}")
            print(f"  • Highly interpretable model")
            
        elif model == 'naive_bayes':
            n_classes = results.get('n_classes', 2)
            print(f"  • Probabilistic classifier")
            print(f"  • Assumes feature independence")
            print(f"  • Fast training and prediction")
            
        elif model == 'logistic_regression':
            c_param = results.get('C', 1.0)
            roc_auc = results.get('roc_auc')
            print(f"  • Linear classifier with regularization")
            print(f"  • C parameter: {c_param}")
            if roc_auc:
                print(f"  • ROC AUC: {roc_auc:.4f}")
                
        elif model == 'knn':
            k_neighbors = results.get('n_neighbors', 5)
            optimal_k = results.get('optimal_k', k_neighbors)
            print(f"  • Instance-based learning")
            print(f"  • Optimal k: {optimal_k}")
            print(f"  • No explicit training phase")

# Generate final summary
print(f"\n" + "=" * 75)
print(f"📋 FINAL SUMMARY - NETWORK INTRUSION DETECTION")
print(f"=" * 75)

best_model_name = best_model['Model']
best_accuracy = best_model['Test_Accuracy']

# Get top 3 features from the best tree-based model
if best_model_name.lower().replace(' ', '_') in ['xgboost', 'random_forest', 'decision_tree']:
    best_features = all_results[best_model_name.lower().replace(' ', '_')]['feature_importance'][:3]
    top_features = [f['feature'] for f in best_features]
else:
    # Use XGBoost features as default if best model is not tree-based
    if 'xgboost' in all_results:
        top_features = [f['feature'] for f in all_results['xgboost']['feature_importance'][:3]]
    elif 'random_forest' in all_results:
        top_features = [f['feature'] for f in all_results['random_forest']['feature_importance'][:3]]
    elif 'decision_tree' in all_results:
        top_features = [f['feature'] for f in all_results['decision_tree']['feature_importance'][:3]]
    else:
        top_features = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts']

print(f"""
Best Model: {best_model_name}
Accuracy: {best_accuracy:.5f}
Top Features: {', '.join(top_features)}

📊 All Models Performance:
{comparison_df[['Model', 'Test_Accuracy', 'F1_Score']].round(4).to_string(index=False)}

💡 Recommendations:
• Use {best_model_name} for deployment (highest accuracy)
• Focus on monitoring: {', '.join(top_features[:2])}
• Consider ensemble methods for even better performance
• Regularly retrain with new network traffic data
""")

# Save comparison results
comparison_df.to_csv('artifacts/model_comparison.csv', index=False)

# Save summary report
summary_report = {
    'best_model': best_model_name,
    'best_accuracy': best_accuracy,
    'top_features': top_features,
    'all_models_performance': comparison_df.to_dict('records'),
    'total_models_trained': len(all_results)
}

with open('artifacts/summary_report.json', 'w') as f:
    json.dump(summary_report, f, indent=2)

print(f"\n✅ Comparison completed! Files saved:")
print("   📁 artifacts/model_comparison.csv")
print("   📁 artifacts/summary_report.json")
print("   📁 plots/model_comparison.png")
print("   📁 plots/all_confusion_matrices.png")

print(f"\n🎉 Network Intrusion Detection Pipeline Completed Successfully!")
print("=" * 75)