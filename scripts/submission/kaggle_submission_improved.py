"""
KAGGLE SUBMISSION - IMPROVED MODEL WITH FEATURE ENGINEERING
===========================================================
Copy this entire code into a Kaggle notebook cell
Requires model files: improved_model_fold_0.pth through improved_model_fold_4.pth
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
import pickle
import gc

import kaggle_evaluation.cmi_inference_server

# ===== CONFIGURATION =====
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_ensemble': True,
    'n_folds': 5,
    
    # Feature engineering flags (must match training)
    'use_angular_velocity': True,
    'use_angular_distance': True,
    'use_statistical_features': True,
    'window_size': 10,
}

# Target BFRB gestures
BFRB_GESTURES = [
    'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
    'Eyebrow - pull hair', 'Eyelash - pull hair', 'Neck - pinch skin',
    'Neck - scratch', 'Cheek - pinch skin'
]


# ===== FEATURE ENGINEERING =====
class FeatureEngineer:
    """Advanced feature engineering for sensor data"""
    
    @staticmethod
    def calculate_angular_velocity(rotation_data, time_diff=0.02):
        """Calculate angular velocity from quaternion data"""
        angular_vel = np.zeros((len(rotation_data) - 1, 3))
        
        for i in range(len(rotation_data) - 1):
            q1 = rotation_data[i]
            q2 = rotation_data[i + 1]
            
            # Quaternion difference
            q_diff = quaternion_multiply(q2, quaternion_conjugate(q1))
            
            # Convert to axis-angle
            angle = 2 * np.arccos(np.clip(q_diff[0], -1, 1))
            if angle > 0:
                axis = q_diff[1:] / np.sin(angle / 2)
                angular_vel[i] = axis * angle / time_diff
            
        # Pad to match original length
        angular_vel = np.vstack([angular_vel, angular_vel[-1]])
        return angular_vel
    
    @staticmethod
    def calculate_angular_distance(rotation_data):
        """Calculate cumulative angular distance traveled"""
        angular_dist = np.zeros(len(rotation_data))
        
        for i in range(1, len(rotation_data)):
            q1 = rotation_data[i - 1]
            q2 = rotation_data[i]
            
            # Quaternion dot product gives cos(theta/2)
            dot = np.clip(np.dot(q1, q2), -1, 1)
            angle = 2 * np.arccos(np.abs(dot))
            angular_dist[i] = angular_dist[i - 1] + angle
            
        return angular_dist
    
    @staticmethod
    def calculate_jerk(acceleration_data, time_diff=0.02):
        """Calculate jerk (derivative of acceleration)"""
        jerk = np.diff(acceleration_data, axis=0) / time_diff
        # Pad to match original length
        jerk = np.vstack([jerk, jerk[-1]])
        return jerk
    
    @staticmethod
    def calculate_mad(data, window_size):
        """Calculate Mean Absolute Deviation over sliding window"""
        mad_features = []
        
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window = data[start_idx:i + 1]
            
            if len(window) > 0:
                mad = np.mean(np.abs(window - np.mean(window, axis=0)), axis=0)
            else:
                mad = np.zeros(data.shape[1])
                
            mad_features.append(mad)
            
        return np.array(mad_features)
    
    @staticmethod
    def engineer_features_polars(df, window_size=10):
        """Apply feature engineering to polars dataframe"""
        # Convert to pandas for easier manipulation
        df_pd = df.to_pandas()
        
        # Extract base features
        acc_data = df_pd[['acc_x', 'acc_y', 'acc_z']].values
        rot_data = df_pd[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
        
        # Angular velocity
        angular_vel = FeatureEngineer.calculate_angular_velocity(rot_data)
        df_pd[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] = angular_vel
        
        # Angular distance
        df_pd['angular_distance'] = FeatureEngineer.calculate_angular_distance(rot_data)
        
        # Jerk
        jerk = FeatureEngineer.calculate_jerk(acc_data)
        df_pd[['jerk_x', 'jerk_y', 'jerk_z']] = jerk
        
        # Acceleration magnitude
        df_pd['acc_magnitude'] = np.linalg.norm(acc_data, axis=1)
        
        # MAD features
        acc_mad = FeatureEngineer.calculate_mad(acc_data, window_size)
        df_pd[['acc_mad_x', 'acc_mad_y', 'acc_mad_z']] = acc_mad
        
        # Rotation angle from quaternion
        df_pd['rotation_angle'] = 2 * np.arccos(np.clip(rot_data[:, 0], -1, 1))
        
        # Convert back to polars
        return pl.from_pandas(df_pd)


# Helper functions for quaternion math
def quaternion_conjugate(q):
    """Return conjugate of quaternion"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


# ===== MODEL ARCHITECTURE =====
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
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
    """Residual block with SE attention"""
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


class ImprovedBFRBModel(nn.Module):
    """Improved model with dynamic feature count"""
    def __init__(self, n_features, n_classes=18):
        super().__init__()
        
        # Calculate feature splits
        self.n_imu_features = 7  # Base IMU features
        self.n_engineered_features = 0
        
        if CONFIG['use_angular_velocity']:
            self.n_engineered_features += 3
        if CONFIG['use_angular_distance']:
            self.n_engineered_features += 1
        if CONFIG['use_statistical_features']:
            self.n_engineered_features += 8  # jerk(3) + acc_mad(3) + acc_mag(1) + rot_angle(1)
            
        self.n_imu_total = self.n_imu_features + self.n_engineered_features
        self.n_thm_features = 5
        self.n_tof_features = n_features - self.n_imu_total - self.n_thm_features
        
        # IMU encoder (now handles engineered features too)
        self.imu_encoder = nn.Sequential(
            nn.Conv1d(self.n_imu_total, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        
        # Thermopile encoder
        self.thm_encoder = nn.Sequential(
            nn.Conv1d(self.n_thm_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),
        )
        
        # Time-of-Flight encoder
        self.tof_encoder = nn.Sequential(
            nn.Conv1d(self.n_tof_features, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        
        # Feature fusion layers
        fusion_dim = 128 + 64 + 128  # IMU + THM + TOF
        self.fusion_layers = nn.Sequential(
            ResidualBlock(fusion_dim, 512),
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool1d(32),  # Dynamic pooling instead of fixed
            ResidualBlock(512, 768),
            ResidualBlock(768, 768),
            nn.AdaptiveAvgPool1d(16),
            ResidualBlock(768, 1024),
        )
        
        # Global context attention
        self.attention = nn.Sequential(
            nn.Conv1d(1024, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        # Binary classification head
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
        
        # Multiclass classification head
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
        # Split sensor data
        imu_data = x[:, :self.n_imu_total, :]
        thm_data = x[:, self.n_imu_total:self.n_imu_total+self.n_thm_features, :]
        tof_data = x[:, self.n_imu_total+self.n_thm_features:, :]
        
        # Encode each sensor type
        imu_features = self.imu_encoder(imu_data)
        thm_features = self.thm_encoder(thm_data)
        tof_features = self.tof_encoder(tof_data)
        
        # Concatenate features
        combined = torch.cat([imu_features, thm_features, tof_features], dim=1)
        
        # Apply fusion layers
        fused = self.fusion_layers(combined)
        
        # Apply attention
        attention_weights = self.attention(fused)
        weighted = fused * attention_weights
        
        # Global pooling
        global_features = weighted.sum(dim=2)
        
        # Shared processing
        shared = self.shared_fc(global_features)
        
        # Task-specific outputs
        binary_output = self.binary_head(shared)
        multiclass_output = self.multiclass_head(shared)
        
        return {
            'binary': binary_output,
            'multiclass': multiclass_output
        }



# ===== GLOBAL PREDICTOR =====
class BFRBPredictor:
    """Main prediction class"""
    
    def __init__(self):
        self.device = torch.device(CONFIG['device'])
        self.models = []
        self.scalers = []
        self.label_encoders = []
        self.feature_cols = None
        
        # Load improved models
        model_paths = [f'/kaggle/input/cmi-models/improved_model_fold_{i}.pth' for i in range(CONFIG['n_folds'])]
        existing_models = [p for p in model_paths if os.path.exists(p)]
        
        if not existing_models:
            raise ValueError("No model files found! Please upload model files to Kaggle.")
        
        print(f"Found {len(existing_models)} model files")
        
        # Load all models for ensemble
        for model_path in existing_models:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load feature columns and model configuration
            if self.feature_cols is None:
                self.feature_cols = checkpoint['feature_cols']
            n_features = checkpoint.get('n_features', len(self.feature_cols))
            
            # Initialize model
            label_encoder = checkpoint.get('label_encoder')
            n_classes = len(label_encoder.classes_) if label_encoder else 18
            
            model = ImprovedBFRBModel(n_features=n_features, n_classes=n_classes)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            self.scalers.append(checkpoint['scaler'])
            self.label_encoders.append(label_encoder)
        
        print(f"Loaded {len(self.models)} models for ensemble")
        
        # Use first label encoder as reference
        self.label_encoder = self.label_encoders[0]
        
        # Create gesture name to ID mapping
        self.gesture_to_id = {name: idx for idx, name in enumerate(self.label_encoder.classes_)}
        
        # BFRB gesture IDs
        self.bfrb_gesture_ids = [self.gesture_to_id[g] for g in BFRB_GESTURES if g in self.gesture_to_id]
    
    def preprocess_sequence(self, df_polars):
        """Preprocess a single sequence"""
        # Apply feature engineering
        df_polars = FeatureEngineer.engineer_features_polars(df_polars, window_size=CONFIG['window_size'])
        
        # Convert to pandas for processing
        df = df_polars.to_pandas()
        
        # Get feature columns
        if self.feature_cols is None:
            # Use default feature columns
            base_feature_cols = [col for col in df.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
            
            engineered_cols = []
            if CONFIG['use_angular_velocity']:
                engineered_cols.extend(['ang_vel_x', 'ang_vel_y', 'ang_vel_z'])
            if CONFIG['use_angular_distance']:
                engineered_cols.append('angular_distance')
            if CONFIG['use_statistical_features']:
                engineered_cols.extend(['jerk_x', 'jerk_y', 'jerk_z', 'acc_magnitude',
                                      'acc_mad_x', 'acc_mad_y', 'acc_mad_z', 'rotation_angle'])
            
            self.feature_cols = base_feature_cols + engineered_cols
        
        # Extract features
        features = df[self.feature_cols].values
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def predict_single(self, features, model_idx=0):
        """Make prediction with a single model"""
        model = self.models[model_idx]
        scaler = self.scalers[model_idx]
        
        # Scale features
        features_scaled = scaler.transform(features)
        features_scaled = np.clip(features_scaled, -10, 10)
        
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(features_scaled.T).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = model(sequence_tensor)
            
            # Get probabilities
            binary_probs = F.softmax(outputs['binary'], dim=1).cpu().numpy()
            multi_probs = F.softmax(outputs['multiclass'], dim=1).cpu().numpy()
            
        return {
            'binary_probs': binary_probs[0],
            'multi_probs': multi_probs[0]
        }
    
    def predict_ensemble(self, features):
        """Make ensemble prediction using all models"""
        all_binary_probs = []
        all_multi_probs = []
        
        # Get predictions from all models
        for i in range(len(self.models)):
            pred = self.predict_single(features, model_idx=i)
            all_binary_probs.append(pred['binary_probs'])
            all_multi_probs.append(pred['multi_probs'])
        
        # Average probabilities
        binary_probs = np.mean(all_binary_probs, axis=0)
        multi_probs = np.mean(all_multi_probs, axis=0)
        
        # Get predictions
        is_bfrb = binary_probs[1] > 0.5
        gesture_id = np.argmax(multi_probs)
        
        # If predicted as BFRB, use the multiclass prediction
        # Otherwise, return the most likely non-BFRB gesture
        if is_bfrb:
            # Check if the predicted gesture is actually a BFRB
            if gesture_id in self.bfrb_gesture_ids:
                final_gesture = self.label_encoder.classes_[gesture_id]
            else:
                # Find the most likely BFRB gesture
                bfrb_probs = [(i, multi_probs[i]) for i in self.bfrb_gesture_ids]
                best_bfrb_id = max(bfrb_probs, key=lambda x: x[1])[0]
                final_gesture = self.label_encoder.classes_[best_bfrb_id]
        else:
            # Find the most likely non-BFRB gesture
            non_bfrb_ids = [i for i in range(len(self.label_encoder.classes_)) 
                           if i not in self.bfrb_gesture_ids]
            if non_bfrb_ids:
                non_bfrb_probs = [(i, multi_probs[i]) for i in non_bfrb_ids]
                best_non_bfrb_id = max(non_bfrb_probs, key=lambda x: x[1])[0]
                final_gesture = self.label_encoder.classes_[best_non_bfrb_id]
            else:
                final_gesture = self.label_encoder.classes_[gesture_id]
        
        return final_gesture
    
    def predict(self, sequence_df):
        """Main prediction function"""
        # Preprocess
        features = self.preprocess_sequence(sequence_df)
        
        # Make prediction
        if CONFIG['use_ensemble'] and len(self.models) > 1:
            return self.predict_ensemble(features)
        else:
            pred = self.predict_single(features, model_idx=0)
            
            # Process single model prediction
            is_bfrb = pred['binary_probs'][1] > 0.5
            gesture_id = np.argmax(pred['multi_probs'])
            
            if is_bfrb and gesture_id in self.bfrb_gesture_ids:
                return self.label_encoder.classes_[gesture_id]
            elif is_bfrb:
                # Find most likely BFRB
                bfrb_probs = [(i, pred['multi_probs'][i]) for i in self.bfrb_gesture_ids]
                best_bfrb_id = max(bfrb_probs, key=lambda x: x[1])[0]
                return self.label_encoder.classes_[best_bfrb_id]
            else:
                # Find most likely non-BFRB
                non_bfrb_ids = [i for i in range(len(self.label_encoder.classes_)) 
                               if i not in self.bfrb_gesture_ids]
                if non_bfrb_ids:
                    non_bfrb_probs = [(i, pred['multi_probs'][i]) for i in non_bfrb_ids]
                    best_non_bfrb_id = max(non_bfrb_probs, key=lambda x: x[1])[0]
                    return self.label_encoder.classes_[best_non_bfrb_id]
                else:
                    return self.label_encoder.classes_[gesture_id]


# ===== INITIALIZE PREDICTOR =====
print("Initializing BFRB predictor...")
predictor = BFRBPredictor()
print("Predictor ready!")


# ===== PREDICTION FUNCTION =====
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Main prediction function for Kaggle evaluation
    
    Args:
        sequence: Polars DataFrame with sensor data for one sequence
        demographics: Polars DataFrame with demographic information
        
    Returns:
        str: Predicted gesture name
    """
    try:
        # Make prediction
        gesture = predictor.predict(sequence)
        return gesture
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Return a default non-BFRB gesture if error
        return 'Text on phone'


# ===== SETUP INFERENCE SERVER =====
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )