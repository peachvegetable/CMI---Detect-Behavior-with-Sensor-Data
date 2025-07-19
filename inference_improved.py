#!/usr/bin/env python3
"""
Improved Inference Script for Kaggle Submission
==============================================
Works with the improved model architecture and feature engineering
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import gc

# Add the evaluation directory to path
sys.path.append('data/kaggle_evaluation')
from cmi_offline_evaluation import CMIEvaluationClient

# ===== CONFIGURATION =====
CONFIG = {
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
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


# ===== FEATURE ENGINEERING (same as training) =====
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
    def engineer_features(df, window_size=10):
        """Apply all feature engineering"""
        engineered_features = []
        
        for seq_id in df['sequence_id'].unique():
            seq_data = df[df['sequence_id'] == seq_id].copy()
            
            # Extract base features
            acc_data = seq_data[['acc_x', 'acc_y', 'acc_z']].values
            rot_data = seq_data[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
            
            # Angular velocity
            angular_vel = FeatureEngineer.calculate_angular_velocity(rot_data)
            seq_data[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] = angular_vel
            
            # Angular distance
            seq_data['angular_distance'] = FeatureEngineer.calculate_angular_distance(rot_data)
            
            # Jerk
            jerk = FeatureEngineer.calculate_jerk(acc_data)
            seq_data[['jerk_x', 'jerk_y', 'jerk_z']] = jerk
            
            # Acceleration magnitude
            seq_data['acc_magnitude'] = np.linalg.norm(acc_data, axis=1)
            
            # MAD features
            acc_mad = FeatureEngineer.calculate_mad(acc_data, window_size)
            seq_data[['acc_mad_x', 'acc_mad_y', 'acc_mad_z']] = acc_mad
            
            # Rotation angle from quaternion
            seq_data['rotation_angle'] = 2 * np.arccos(np.clip(rot_data[:, 0], -1, 1))
            
            engineered_features.append(seq_data)
            
        return pd.concat(engineered_features, ignore_index=True)


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


# ===== MODEL ARCHITECTURE (must match training) =====
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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


# ===== INFERENCE DATASET =====
class InferenceDataset(Dataset):
    """Dataset for inference without labels"""
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return transposed sequence for Conv1d (channels, time)
        return torch.FloatTensor(self.sequences[idx].T)


# ===== PREDICTION CLASS =====
class BFRBPredictor:
    """Main prediction class for Kaggle submission"""
    
    def __init__(self, model_paths, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.scalers = []
        self.label_encoders = []
        self.feature_cols = None
        
        # Load all models for ensemble
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Loading model: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Extract configuration
                if self.feature_cols is None:
                    self.feature_cols = checkpoint.get('feature_cols')
                    n_features = checkpoint.get('n_features', len(self.feature_cols) if self.feature_cols else 332)
                
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
                
        if not self.models:
            raise ValueError("No models loaded!")
            
        print(f"Loaded {len(self.models)} models for ensemble")
        
        # Use first label encoder as reference
        self.label_encoder = self.label_encoders[0]
        
        # Create gesture name to ID mapping
        self.gesture_to_id = {name: idx for idx, name in enumerate(self.label_encoder.classes_)}
        
        # BFRB gesture IDs
        self.bfrb_gesture_ids = [self.gesture_to_id[g] for g in BFRB_GESTURES if g in self.gesture_to_id]
    
    def preprocess_sequence(self, df):
        """Preprocess a single sequence"""
        # Apply feature engineering
        if CONFIG['use_angular_velocity'] or CONFIG['use_angular_distance'] or CONFIG['use_statistical_features']:
            df = FeatureEngineer.engineer_features(df, window_size=CONFIG['window_size'])
        
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
        
        return {
            'gesture': final_gesture,
            'binary_probs': binary_probs,
            'multi_probs': multi_probs,
            'confidence': max(multi_probs)
        }
    
    def predict_sequence(self, df):
        """Predict gesture for a sequence dataframe"""
        # Preprocess
        features = self.preprocess_sequence(df)
        
        # Make prediction
        if CONFIG['use_ensemble'] and len(self.models) > 1:
            return self.predict_ensemble(features)
        else:
            pred = self.predict_single(features, model_idx=0)
            
            # Process single model prediction
            is_bfrb = pred['binary_probs'][1] > 0.5
            gesture_id = np.argmax(pred['multi_probs'])
            
            if is_bfrb and gesture_id in self.bfrb_gesture_ids:
                final_gesture = self.label_encoder.classes_[gesture_id]
            elif is_bfrb:
                # Find most likely BFRB
                bfrb_probs = [(i, pred['multi_probs'][i]) for i in self.bfrb_gesture_ids]
                best_bfrb_id = max(bfrb_probs, key=lambda x: x[1])[0]
                final_gesture = self.label_encoder.classes_[best_bfrb_id]
            else:
                # Find most likely non-BFRB
                non_bfrb_ids = [i for i in range(len(self.label_encoder.classes_)) 
                               if i not in self.bfrb_gesture_ids]
                if non_bfrb_ids:
                    non_bfrb_probs = [(i, pred['multi_probs'][i]) for i in non_bfrb_ids]
                    best_non_bfrb_id = max(non_bfrb_probs, key=lambda x: x[1])[0]
                    final_gesture = self.label_encoder.classes_[best_non_bfrb_id]
                else:
                    final_gesture = self.label_encoder.classes_[gesture_id]
            
            return {
                'gesture': final_gesture,
                'binary_probs': pred['binary_probs'],
                'multi_probs': pred['multi_probs'],
                'confidence': max(pred['multi_probs'])
            }


# ===== MAIN EXECUTION =====
def main():
    print("IMPROVED BFRB DETECTION - KAGGLE SUBMISSION")
    print("=" * 60)
    
    # Model paths
    model_paths = [f'improved_model_fold_{i}.pth' for i in range(CONFIG['n_folds'])]
    
    # Filter existing models
    existing_models = [p for p in model_paths if os.path.exists(p)]
    
    if not existing_models:
        print("ERROR: No trained models found!")
        print("Please run train_improved.py first.")
        return
    
    print(f"Found {len(existing_models)} trained models")
    
    # Initialize predictor
    predictor = BFRBPredictor(existing_models, device=CONFIG['device'])
    
    # Initialize evaluation client
    print("\nInitializing evaluation client...")
    client = CMIEvaluationClient()
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    
    with tqdm(total=len(client.test_sequence_ids)) as pbar:
        for sequence_id in client.test_sequence_ids:
            # Get sequence data
            sequence_df = client.get_test_sequence(sequence_id)
            
            # Make prediction
            result = predictor.predict_sequence(sequence_df)
            
            predictions.append({
                'sequence_id': sequence_id,
                'gesture': result['gesture']
            })
            
            pbar.update(1)
            pbar.set_postfix({
                'gesture': result['gesture'],
                'conf': f"{result['confidence']:.3f}"
            })
    
    # Create submission dataframe
    submission_df = pd.DataFrame(predictions)
    
    # Submit predictions
    print("\nSubmitting predictions...")
    score = client.submit_predictions(submission_df)
    
    print("\n" + "="*60)
    print(f"Submission complete!")
    print(f"Score: {score}")
    print("="*60)
    
    # Save predictions locally
    submission_df.to_csv('improved_test_predictions.csv', index=False)
    print(f"\nPredictions saved to: improved_test_predictions.csv")
    
    # Save detailed results
    results = {
        'score': score,
        'n_models': len(existing_models),
        'config': CONFIG,
        'predictions_sample': predictions[:10]  # Save first 10 for inspection
    }
    
    with open('improved_submission_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: improved_submission_results.json")


if __name__ == "__main__":
    main()