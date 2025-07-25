# Decision Tree Model for Network Intrusion Detection
# Notebook: 06_decision_tree_model.ipynb

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import warnings
warnings.filterwarnings('ignore')

print("🌳 Decision Tree Model Training for Network Intrusion Detection")
print("=" * 70)

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

# Initialize Decision Tree model
print(f"\n🌳 Initializing Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

# 5-fold cross-validation
print(f"\n🔄 Performing 5-fold cross-validation...")
cv_scores = cross_val_score(
    dt_model, X_train, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

print(f"CV Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train the model
print(f"\n🏋️ Training Decision Tree model on full training set...")
dt_model.fit(X_train, y_train)
print("✅ Training completed!")

# Make predictions
print(f"\n🔮 Making predictions on test set...")
y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)

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

# Feature importances (built-in for Decision Trees)
print(f"\n🏆 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': feature_list,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Decision Tree specific information
print(f"\n🌳 Decision Tree Information:")
print(f"Tree depth: {dt_model.get_depth()}")
print(f"Number of leaves: {dt_model.get_n_leaves()}")
print(f"Number of nodes: {dt_model.tree_.node_count}")
print(f"Criterion: {dt_model.criterion}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.title('Decision Tree - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/confusion_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize feature importances
plt.figure(figsize=(10, 6))
feature_importance_plot = feature_importance.head(10)
plt.barh(range(len(feature_importance_plot)), feature_importance_plot['importance'])
plt.yticks(range(len(feature_importance_plot)), feature_importance_plot['feature'])
plt.xlabel('Feature Importance')
plt.title('Decision Tree - Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/feature_importance_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize the decision tree (simplified version)
plt.figure(figsize=(20, 12))
plot_tree(dt_model, 
          feature_names=feature_list,
          class_names=['Benign', 'Malicious'],
          filled=True,
          max_depth=3,  # Show only top 3 levels for clarity
          fontsize=10)
plt.title('Decision Tree Structure (Top 3 Levels)')
plt.tight_layout()
plt.savefig('plots/decision_tree_structure.png', dpi=300, bbox_inches='tight')
plt.show()

# Decision rules extraction (top rules)
print(f"\n📋 Top Decision Rules:")
tree = dt_model.tree_
feature_names = feature_list

def get_rules(tree, feature_names, node=0, depth=0, rule=""):
    if depth > 2:  # Limit depth for readability
        return []
    
    rules = []
    if tree.children_left[node] != tree.children_right[node]:  # Not a leaf
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        
        # Left child (<=)
        left_rule = f"{rule}IF {feature} <= {threshold:.3f}"
        rules.extend(get_rules(tree, feature_names, tree.children_left[node], depth+1, left_rule + " AND "))
        
        # Right child (>)
        right_rule = f"{rule}IF {feature} > {threshold:.3f}"
        rules.extend(get_rules(tree, feature_names, tree.children_right[node], depth+1, right_rule + " AND "))
    else:
        # Leaf node
        samples = tree.n_node_samples[node]
        value = tree.value[node][0]
        prediction = "Malicious" if np.argmax(value) == 1 else "Benign"
        confidence = max(value) / samples
        rules.append(f"{rule}THEN {prediction} (confidence: {confidence:.3f}, samples: {samples})")
    
    return rules

rules = get_rules(tree, feature_names)
for i, rule in enumerate(rules[:5], 1):  # Show top 5 rules
    print(f"{i}. {rule}")

# Save the model
print(f"\n💾 Saving Decision Tree model...")
joblib.dump(dt_model, 'artifacts/model_decision_tree.pkl')

# Save evaluation results
results = {
    'model_name': 'Decision_Tree',
    'accuracy': float(accuracy),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'cv_scores': cv_scores.tolist(),
    'classification_report': class_report,
    'confusion_matrix': cm.tolist(),
    'feature_importance': feature_importance.to_dict('records'),
    'tree_depth': int(dt_model.get_depth()),
    'n_leaves': int(dt_model.get_n_leaves()),
    'n_nodes': int(dt_model.tree_.node_count),
    'criterion': dt_model.criterion,
    'max_depth': dt_model.max_depth,
    'min_samples_split': dt_model.min_samples_split,
    'min_samples_leaf': dt_model.min_samples_leaf
}

with open('artifacts/results_decision_tree.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Saved successfully:")
print("   📁 artifacts/model_decision_tree.pkl")
print("   📁 artifacts/results_decision_tree.json")
print("   📁 plots/confusion_decision_tree.png")
print("   📁 plots/feature_importance_decision_tree.png")
print("   📁 plots/decision_tree_structure.png")

print(f"\n🎉 Decision Tree model training completed!")
print(f"Final Test Accuracy: {accuracy:.4f}")
print(f"Tree Depth: {dt_model.get_depth()}")
print("=" * 70)