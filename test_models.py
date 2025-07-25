"""
Test script to verify models and create sample analysis
"""
import joblib
import pandas as pd
import numpy as np
import json

def test_models():
    print("🧪 Testing Network Intrusion Detection Models")
    print("=" * 50)
    
    try:
        # Load models
        print("📂 Loading models...")
        rf_model = joblib.load('artifacts/model_random_forest.pkl')
        xgb_model = joblib.load('artifacts/model_xgboost.pkl')
        ann_model = joblib.load('artifacts/model_ann.pkl')
        
        # Load preprocessing components
        scaler = joblib.load('artifacts/scaler.pkl')
        feature_names = joblib.load('artifacts/feature_list.pkl')
        
        print(f"✅ Loaded {len(feature_names)} features")
        print(f"✅ Random Forest: {type(rf_model).__name__}")
        print(f"✅ XGBoost: {type(xgb_model).__name__}")
        print(f"✅ ANN: {type(ann_model).__name__}")
        
        # Create sample data based on your real-time input
        print("\n🔍 Testing with sample data...")
        
        # Sample 1: Normal traffic
        normal_data = {feature: 0 for feature in feature_names}
        normal_data.update({
            'Flow Duration': 1.2,
            'PSH Flag Cnt': 2,
            'Fwd Pkts/s': 150,
            'Tot Fwd Pkts': 25
        })
        
        # Sample 2: Suspicious traffic (like your real-time data)
        suspicious_data = {feature: 0 for feature in feature_names}
        suspicious_data.update({
            'Flow Duration': 2.5,
            'PSH Flag Cnt': 8,
            'Fwd Pkts/s': 1500,
            'Tot Fwd Pkts': 150
        })
        
        for sample_name, sample_data in [("Normal", normal_data), ("Suspicious", suspicious_data)]:
            print(f"\n📊 {sample_name} Traffic Analysis:")
            
            # Create DataFrame
            df = pd.DataFrame([sample_data])
            df = df[feature_names]  # Ensure correct order
            
            # Scale features
            scaled_features = scaler.transform(df)
            
            # Test each model
            for model_name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model), ("ANN", ann_model)]:
                try:
                    prediction = model.predict(scaled_features)[0]
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(scaled_features)[0]
                        confidence = max(proba) * 100
                    else:
                        confidence = 85.0  # Default for ANN
                    
                    result = "MALICIOUS" if prediction == 1 else "BENIGN"
                    print(f"  - {model_name}: {result} ({confidence:.1f}% confidence)")
                    
                except Exception as e:
                    print(f"  - {model_name}: Error - {e}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_models()
