"""
Network Intrusion Detection Web Application
Real-time network traffic analysis and threat detection
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import time
import random

app = Flask(__name__)

class NetworkIntrusionDetector:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.load_models()
        
    def load_models(self):
        """Load all trained models and preprocessing components"""
        try:
            # Load the best performing model (Random Forest)
            self.models['random_forest'] = joblib.load('artifacts/model_random_forest.pkl')
            self.models['xgboost'] = joblib.load('artifacts/model_xgboost.pkl')
            self.models['ann'] = joblib.load('artifacts/model_ann.pkl')
            
            # Load scaler and feature names
            self.scaler = joblib.load('artifacts/scaler.pkl')
            with open('artifacts/feature_list.pkl', 'rb') as f:
                self.feature_names = joblib.load(f)
                
            print("✅ Models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def predict_traffic(self, packet_data, model_name='random_forest'):
        """Predict if network traffic is benign or malicious"""
        try:
            # Create feature vector with all required features
            feature_vector = {}
            
            # Map input data to actual feature names
            feature_vector['Flow Duration'] = packet_data.get('flow_duration', 2.5)
            feature_vector['PSH Flag Cnt'] = packet_data.get('psh_flags', 8)
            feature_vector['Fwd Pkts/s'] = packet_data.get('packet_rate', 1500)
            feature_vector['Tot Fwd Pkts'] = packet_data.get('total_packets', random.randint(50, 200))
            
            # Add other required features with reasonable defaults
            for feature in self.feature_names:
                if feature not in feature_vector:
                    if 'pkts' in feature.lower() or 'packets' in feature.lower():
                        feature_vector[feature] = random.randint(1, 100)
                    elif 'byte' in feature.lower() or 'len' in feature.lower():
                        feature_vector[feature] = random.randint(20, 1500)
                    elif 'flag' in feature.lower():
                        feature_vector[feature] = random.randint(0, 5)
                    elif 'time' in feature.lower() or 'duration' in feature.lower():
                        feature_vector[feature] = random.uniform(0.1, 5.0)
                    else:
                        feature_vector[feature] = 0
            
            # Create DataFrame with correct feature order
            df = pd.DataFrame([feature_vector])
            df = df[self.feature_names]  # Ensure correct order
            
            # Scale the features
            scaled_features = self.scaler.transform(df)
            
            # Get prediction
            model = self.models[model_name]
            prediction = model.predict(scaled_features)[0]
            prediction_proba = model.predict_proba(scaled_features)[0]
            
            confidence = max(prediction_proba) * 100
            
            # Adjust prediction based on suspicious patterns
            if packet_data.get('psh_flags', 0) > 8 and packet_data.get('packet_rate', 0) > 1500:
                # High PSH flags + high packet rate = likely DDoS/flood attack
                confidence = min(99.7, confidence + 10)
                prediction = 1
            
            result = {
                'prediction': 'MALICIOUS' if prediction == 1 else 'BENIGN',
                'confidence': round(confidence, 2),
                'threat_level': self.get_threat_level(confidence, prediction),
                'model_used': model_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_threat_level(self, confidence, prediction):
        """Determine threat level based on prediction and confidence"""
        if prediction == 0:  # Benign
            return 'LOW'
        else:  # Malicious
            if confidence >= 95:
                return 'CRITICAL'
            elif confidence >= 85:
                return 'HIGH'
            elif confidence >= 70:
                return 'MEDIUM'
            else:
                return 'LOW'

# Initialize detector
detector = NetworkIntrusionDetector()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_packet():
    """Analyze a network packet"""
    try:
        data = request.get_json()
        
        # Use the provided real-time data structure
        packet_features = {
            'flow_duration': data.get('flow_duration', 2.5),
            'psh_flags': data.get('psh_flags', 8),
            'packet_rate': data.get('packet_rate', 1500),
            'total_packets': data.get('total_packets', random.randint(50, 200)),
        }
        
        model_name = data.get('model', 'random_forest')
        result = detector.predict_traffic(packet_features, model_name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate')
def simulate_traffic():
    """Simulate real-time network traffic data"""
    # Generate realistic network packet data
    packet_data = {
        'flow_duration': round(random.uniform(0.1, 5.0), 2),
        'psh_flags': random.randint(0, 15),
        'packet_rate': random.randint(100, 3000),
        'payload_size': random.choice(['Normal', 'Unusual pattern', 'Suspicious', 'Large']),
        'protocol': random.choice(['TCP', 'UDP', 'ICMP']),
        'source_ip': f"192.168.1.{random.randint(1, 254)}",
        'dest_ip': f"10.0.0.{random.randint(1, 254)}",
        'source_port': random.randint(1024, 65535),
        'dest_port': random.choice([80, 443, 22, 21, 25, 53, random.randint(1024, 65535)]),
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }
    
    return jsonify(packet_data)

@app.route('/api/stats')
def get_stats():
    """Get model performance statistics"""
    try:
        with open('artifacts/summary_report.json', 'r') as f:
            stats = json.load(f)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
