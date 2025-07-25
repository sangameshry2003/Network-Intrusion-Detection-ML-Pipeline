# Network Intrusion Detection - Data Preprocessing
# Notebook: 00_data_preprocessing.ipynb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    import cudf
    import cupy as cp
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.model_selection import train_test_split as cu_train_test_split
    GPU_AVAILABLE = True
    print("🚀 RAPIDS cuDF/cuML detected: Using GPU acceleration.")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ RAPIDS not found: Falling back to CPU (pandas/sklearn).")

# Create directories
os.makedirs('artifacts', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("🔧 Stage 1: Data Preprocessing for Network Intrusion Detection")
print("=" * 60)

# Load dataset
print("📂 Loading dataset...")
if GPU_AVAILABLE:
    df = cudf.read_csv('processed_friday_dataset.csv')
else:
    df = pd.read_csv('processed_friday_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")
print(f"Samples: {df.shape[0]}")

# Display basic info
print("\n📊 Dataset Info:")
print(df.info())

# Check label distribution
print(f"\n🏷️ Label Distribution:")
print(df['Label'].value_counts())

# Note: Current dataset has only benign traffic (Label=0)
# For demonstration, let's create some synthetic malicious samples
print("\n⚠️ Note: Current dataset contains only benign traffic (Label=0)")
print("Creating synthetic malicious samples for balanced dataset...")

# Create synthetic malicious samples by modifying existing data
np.random.seed(42)
malicious_samples = []

# Generate malicious samples with different characteristics
for i in range(len(df) // 2):  # Create half as many malicious samples
    sample = df.sample(1, replace=True).copy()
    
    # Modify characteristics to simulate malicious behavior
    sample.iloc[0, df.columns.get_loc('Flow Duration')] = int(sample.iloc[0, df.columns.get_loc('Flow Duration')] * np.random.uniform(2, 5))
    sample.iloc[0, df.columns.get_loc('Tot Fwd Pkts')] = int(sample.iloc[0, df.columns.get_loc('Tot Fwd Pkts')] * np.random.uniform(3, 10))
    sample.iloc[0, df.columns.get_loc('Tot Bwd Pkts')] = int(sample.iloc[0, df.columns.get_loc('Tot Bwd Pkts')] * np.random.uniform(3, 10))
    sample.iloc[0, df.columns.get_loc('Fwd Pkts/s')] *= np.random.uniform(5, 20)
    sample.iloc[0, df.columns.get_loc('PSH Flag Cnt')] = np.random.randint(3, 8)
    sample.iloc[0, df.columns.get_loc('RST Flag Cnt')] = np.random.randint(2, 5)
    sample['Label'] = 1  # Malicious
    
    malicious_samples.append(sample)

# Combine original and synthetic data
malicious_df = pd.concat(malicious_samples, ignore_index=True)
df_balanced = pd.concat([df, malicious_df], ignore_index=True)

print(f"\n📈 Enhanced Dataset:")
print(f"Total samples: {len(df_balanced)}")
print(f"Label distribution:")
print(df_balanced['Label'].value_counts())

# Select 10 specific features for modeling
selected_features = [
    'Flow Duration',
    'Tot Fwd Pkts', 
    'Tot Bwd Pkts',
    'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean',
    'Flow IAT Mean',
    'Fwd Pkts/s',
    'Bwd Pkts/s',
    'PSH Flag Cnt',
    'ACK Flag Cnt'
]

print(f"\n🎯 Selected Features ({len(selected_features)}):")
for i, feature in enumerate(selected_features, 1):
    print(f"{i:2d}. {feature}")

# Prepare features and labels
if GPU_AVAILABLE:
    X = df_balanced[selected_features].astype('float32')
    y = df_balanced['Label'].astype('int32')
else:
    X = df_balanced[selected_features]
    y = df_balanced['Label']

print(f"\n📋 Feature Matrix Shape: {X.shape}")
print(f"Labels Shape: {y.shape}")

# Check for missing values
print(f"\n🔍 Missing Values Check:")
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print(missing_counts[missing_counts > 0])
else:
    print("✅ No missing values found")

# Feature scaling
print(f"\n⚖️ Applying StandardScaler...")
if GPU_AVAILABLE:
    scaler = cuStandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = cudf.DataFrame(X_scaled, columns=selected_features)
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)

print("Scaling completed successfully!")

# Stratified train-test split (80:20)
print(f"\n🔄 Performing stratified train-test split (80:20)...")
if GPU_AVAILABLE:
    X_train, X_test, y_train, y_test = cu_train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training label distribution:")
print(y_train.value_counts())
print(f"Test label distribution:")
print(y_test.value_counts())

# Save preprocessing artifacts
print(f"\n💾 Saving preprocessing artifacts...")

# Save train-test split
joblib.dump(X_train, 'artifacts/X_train.pkl')
joblib.dump(X_test, 'artifacts/X_test.pkl')
joblib.dump(y_train, 'artifacts/y_train.pkl')
joblib.dump(y_test, 'artifacts/y_test.pkl')

# Save feature list and scaler
joblib.dump(selected_features, 'artifacts/feature_list.pkl')
joblib.dump(scaler, 'artifacts/scaler.pkl')

# Save original and enhanced datasets
df.to_csv('artifacts/original_dataset.csv', index=False)
df_balanced.to_csv('artifacts/enhanced_dataset.csv', index=False)

print("✅ Saved successfully:")
print("   📁 artifacts/X_train.pkl")
print("   📁 artifacts/X_test.pkl") 
print("   📁 artifacts/y_train.pkl")
print("   📁 artifacts/y_test.pkl")
print("   📁 artifacts/feature_list.pkl")
print("   📁 artifacts/scaler.pkl")
print("   📁 artifacts/original_dataset.csv")
print("   📁 artifacts/enhanced_dataset.csv")

# Display feature statistics
print(f"\n📊 Feature Statistics (Scaled):")
print(X_train.describe().round(3))

print(f"\n🎉 Data preprocessing completed successfully!")
print(f"Ready for Stage 2: Model Training")
print("=" * 60)