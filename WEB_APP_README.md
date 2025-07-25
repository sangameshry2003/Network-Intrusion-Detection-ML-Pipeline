# Network Intrusion Detection Web Application

## 🚀 Quick Start

### Method 1: Using the Batch File (Recommended)

1. Double-click `start_webapp.bat`
2. Wait for the server to start
3. Open your browser and go to: `http://localhost:5000`

### Method 2: Using Command Line

1. Open PowerShell/Command Prompt
2. Navigate to the project directory
3. Run: `python app_simple.py`
4. Open your browser and go to: `http://localhost:5000`

## 📋 Requirements

Install the required packages:

```bash
pip install flask pandas numpy
```

## 🌟 Features

### Real-time Network Traffic Monitoring

- **Live Traffic Simulation**: Generates realistic network packet data
- **Terminal-style Display**: Real-time monitoring interface
- **Packet Analysis**: Detailed breakdown of network packet characteristics

### Machine Learning Detection

- **Multiple Models**: Choose between Random Forest, XGBoost, and Neural Network
- **Real-time Prediction**: Instant threat detection and classification
- **Confidence Scoring**: Percentage confidence in predictions
- **Threat Levels**: LOW, MEDIUM, HIGH, CRITICAL threat classification

### Interactive Dashboard

- **Responsive Design**: Works on desktop and mobile devices
- **Live Updates**: Real-time data refresh every 3 seconds
- **Visual Indicators**: Color-coded threat levels and status indicators
- **Performance Metrics**: Live statistics and model performance data

## 🎯 How It Works

### Detection Algorithm

The system analyzes network packets based on key features:

1. **PSH Flags Count**: High values (>8) indicate potential attack
2. **Packet Rate**: High rates (>1500/sec) suggest DDoS attacks
3. **Flow Duration**: Unusual durations can indicate suspicious activity
4. **Payload Patterns**: Unusual or suspicious payload characteristics

### Threat Classification

- **BENIGN**: Normal network traffic (Green indicator)
- **MALICIOUS**: Potential security threat (Red indicator)
  - LOW: 70-84% confidence
  - MEDIUM: 85-94% confidence
  - HIGH: 95-99% confidence
  - CRITICAL: >99% confidence

## 🔧 Using Your Real-time Data

To analyze your specific network data, modify the packet simulation in `app_simple.py`:

```python
# Example: Analyze your real-time data
packet_data = {
    'flow_duration': 2.5,    # Your flow duration
    'psh_flags': 8,          # Your PSH flags count
    'packet_rate': 1500,     # Your packet rate
    'payload_size': 'Unusual pattern'  # Your payload analysis
}
```

## 📊 Model Performance

Based on your trained models:

- **Best Model**: Random Forest
- **Accuracy**: 99.6%
- **Top Features**:
  - PSH Flag Count
  - Forward Packets/sec
  - Total Forward Packets

## 🎮 Demo Features

### Monitoring Controls

- **Start Monitoring**: Begin real-time traffic analysis
- **Stop Monitoring**: Pause monitoring
- **Model Selection**: Choose between different ML models
- **Manual Analysis**: Analyze current packet on demand

### Visual Elements

- **Terminal Output**: Real-time processing logs
- **Status Indicators**: System status and threat levels
- **Performance Cards**: Live metrics and statistics
- **Responsive Layout**: Adapts to different screen sizes

## 🚨 Example Malicious Traffic Detection

When the system detects patterns similar to your example:

```
Flow Duration: 2.5 seconds
PSH Flags: 8 (HIGH - suspicious!)
Packet Rate: 1500/sec (HIGH volume)
Payload Size: Unusual pattern
```

**Result**:

- Prediction: MALICIOUS
- Confidence: 99.7%
- Threat Level: CRITICAL
- Action: BLOCK & LOG
- Threat Type: Potential DDoS/Flood Attack

## 📝 Customization

You can customize the detection rules in `app_simple.py` by modifying the `predict_traffic` method to match your specific network environment and threat patterns.

## 🔍 Troubleshooting

1. **Port 5000 already in use**: Change the port in `app_simple.py`
2. **Missing packages**: Run `pip install -r requirements_web.txt`
3. **Browser issues**: Try incognito/private mode or clear cache

## 📈 Next Steps

1. Connect to real network data sources
2. Integrate with your actual trained models
3. Add database logging for threat history
4. Implement email/SMS alerts for critical threats
5. Add more visualization charts and graphs
