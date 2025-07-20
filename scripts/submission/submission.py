#!/usr/bin/env python3
"""
Kaggle Submission Script for CMI Competition
============================================
This script interfaces with the Kaggle evaluation API
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import polars as pl

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define the model architecture that matches the saved models
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout(0.1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BestBFRBModel(nn.Module):
    """Model architecture matching the saved best_model_fold_*.pth files"""
    def __init__(self, n_classes=18):
        super().__init__()
        
        # Sensor encoders
        self.imu_encoder = nn.Sequential(
            nn.Conv1d(7, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        
        self.thm_encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),
        )
        
        self.tof_encoder = nn.Sequential(
            nn.Conv1d(320, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            ResidualBlock(320, 512),
            ResidualBlock(512, 512),
            nn.MaxPool1d(2),
            ResidualBlock(512, 768),
            ResidualBlock(768, 768),
            nn.MaxPool1d(2),
            ResidualBlock(768, 1024),
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Conv1d(1024, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Shared FC
        self.shared_fc = nn.Sequential(
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        # Task-specific pathways (matching saved model names)
        self.binary_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(192, 2)
        )
        
        self.multiclass_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        # Split sensors
        imu_data = x[:, :7, :]
        thm_data = x[:, 7:12, :]
        tof_data = x[:, 12:, :]
        
        # Encode
        imu_features = self.imu_encoder(imu_data)
        thm_features = self.thm_encoder(thm_data)
        tof_features = self.tof_encoder(tof_data)
        
        # Concatenate
        combined = torch.cat([imu_features, thm_features, tof_features], dim=1)
        
        # Fusion
        fused = self.fusion_layers(combined)
        
        # Attention
        attention_weights = self.attention(fused)
        weighted = fused * attention_weights
        global_features = weighted.sum(dim=2)
        
        # Shared processing
        shared = self.shared_fc(global_features)
        
        # Task outputs
        binary_output = self.binary_head(shared)
        multiclass_output = self.multiclass_head(shared)
        
        return {
            'binary': binary_output,
            'multiclass': multiclass_output
        }

# Configuration
WINDOW_SIZE = 64  # Must match the training window size from train_best.py
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gesture classes (in correct order)
GESTURE_CLASSES = [
    'Above ear - pull hair', 'Cheek - pinch skin', 'Drink from bottle/cup',
    'Eyebrow - pull hair', 'Eyelash - pull hair', 
    'Feel around in tray and pull out an object', 'Forehead - pull hairline',
    'Forehead - scratch', 'Glasses on/off', 'Neck - pinch skin',
    'Neck - scratch', 'Pinch knee/leg skin', 'Pull air toward your face',
    'Scratch knee/leg skin', 'Text on phone', 'Wave hello',
    'Write name in air', 'Write name on leg'
]

# Feature columns
IMU_FEATURES = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
THM_FEATURES = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
TOF_FEATURES = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
ALL_FEATURES = IMU_FEATURES + THM_FEATURES + TOF_FEATURES

# Load models and scalers at startup
print("Loading models...")
models = []
scalers = []

# Create default scaler if not in checkpoint
default_scaler = StandardScaler()

for fold in range(5):
    # Try best_model_fold first (new training script)
    model_path = f'best_model_fold_{fold}.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model = BestBFRBModel(n_classes=18).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        
        # Get scaler from checkpoint or use default
        if 'scaler' in checkpoint:
            scalers.append(checkpoint['scaler'])
        else:
            # Create a standard scaler as default
            scalers.append(default_scaler)
            
        print(f"✅ Loaded fold {fold}")
    else:
        # Try alternative model names for backward compatibility
        alt_path = f'hybrid_model_fold_{fold}.pth'
        if os.path.exists(alt_path):
            checkpoint = torch.load(alt_path, map_location=DEVICE, weights_only=False)
            model = BestBFRBModel(n_classes=18).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
            
            if 'scaler' in checkpoint:
                scalers.append(checkpoint['scaler'])
            else:
                scalers.append(default_scaler)
                
            print(f"✅ Loaded fold {fold} (alternative)")

if not models:
    raise RuntimeError("No trained models found! Please ensure model files are in the current directory.")

print(f"Loaded {len(models)} models")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame = None) -> str:
    """
    Make prediction for a single sequence
    
    Args:
        sequence: Polars DataFrame with sensor data
        demographics: Polars DataFrame with participant info (unused)
    
    Returns:
        str: Predicted gesture name
    """
    try:
        # Convert to pandas for processing
        sequence_df = sequence.to_pandas()
        
        # Extract features
        data = sequence_df[ALL_FEATURES].values.astype(np.float32)
        
        # Handle missing values
        # For TOF: -1 means no object detected (keep as is)
        # For other sensors: fill with 0
        for i, col in enumerate(ALL_FEATURES):
            col_data = data[:, i]
            if col in TOF_FEATURES:
                # TOF: replace NaN with 0, keep -1 as is
                data[:, i] = np.where(np.isnan(col_data), 0.0, col_data)
            else:
                # Other sensors: replace NaN with 0
                data[:, i] = np.where(np.isnan(col_data), 0.0, col_data)
        
        # Create windows and make predictions
        all_predictions = []
        
        for model, scaler in zip(models, scalers):
            # Create window
            if len(data) >= WINDOW_SIZE:
                # Use the last window
                window = data[-WINDOW_SIZE:]
            else:
                # Pad if too short
                padding = WINDOW_SIZE - len(data)
                window = np.pad(data, ((0, padding), (0, 0)), mode='constant', constant_values=0)
            
            # Normalize
            if hasattr(scaler, 'scale_'):
                # Scaler is fitted
                window_scaled = scaler.transform(window)
            else:
                # Scaler not fitted, fit it on the window data
                window_scaled = scaler.fit_transform(window)
            
            # Convert to tensor (channels, time)
            window_tensor = torch.FloatTensor(window_scaled.T).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs = model(window_tensor)
                multi_probs = F.softmax(outputs['multiclass'], dim=1)
                all_predictions.append(multi_probs.cpu().numpy())
        
        # Average predictions across models
        avg_predictions = np.mean(all_predictions, axis=0)
        predicted_idx = np.argmax(avg_predictions[0])
        
        # Get gesture name
        predicted_gesture = GESTURE_CLASSES[predicted_idx]
        
        return predicted_gesture
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Return most common non-BFRB gesture as fallback
        return "Text on phone"


# The predict function is what Kaggle will call
print("Submission script ready!")
print(f"Device: {DEVICE}")
print(f"Number of models: {len(models)}")