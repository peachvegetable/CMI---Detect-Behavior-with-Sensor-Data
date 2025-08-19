import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R
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
    def remove_gravity_from_acc(acc_data, rot_data):
        """Remove gravity from accelerometer data using quaternion rotation"""
        if isinstance(acc_data, pd.DataFrame):
            acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
        else:
            acc_values = acc_data
        
        if isinstance(rot_data, pd.DataFrame):
            # Note: rot_data columns are [rot_w, rot_x, rot_y, rot_z]
            # scipy expects [x, y, z, w] format
            quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
        else:
            # Reorder from [w, x, y, z] to [x, y, z, w]
            quat_values = rot_data[:, [1, 2, 3, 0]]
        
        num_samples = acc_values.shape[0]
        linear_accel = np.zeros_like(acc_values)
        gravity_world = np.array([0, 0, 9.81])
        
        for i in range(num_samples):
            if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
                linear_accel[i, :] = acc_values[i, :]
                continue
            
            try:
                rotation = R.from_quat(quat_values[i])
                gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
                linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
            except ValueError:
                linear_accel[i, :] = acc_values[i, :]
        
        return linear_accel
    
    @staticmethod
    def calculate_angular_velocity(rotation_data, time_diff=0.02):
        """Calculate angular velocity from quaternion data using scipy"""
        if isinstance(rotation_data, pd.DataFrame):
            quat_values = rotation_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
        else:
            # Reorder from [w, x, y, z] to [x, y, z, w] for scipy
            quat_values = rotation_data[:, [1, 2, 3, 0]]
        
        num_samples = quat_values.shape[0]
        angular_vel = np.zeros((num_samples, 3))
        
        for i in range(num_samples - 1):
            q_t = quat_values[i]
            q_t_plus_dt = quat_values[i + 1]
            
            if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
               np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
                continue
            
            try:
                rot_t = R.from_quat(q_t)
                rot_t_plus_dt = R.from_quat(q_t_plus_dt)
                
                # Calculate relative rotation
                delta_rot = rot_t.inv() * rot_t_plus_dt
                
                # Convert to angular velocity
                angular_vel[i, :] = delta_rot.as_rotvec() / time_diff
            except ValueError:
                pass
        
        return angular_vel
    
    @staticmethod
    def calculate_angular_distance(rotation_data):
        """Calculate frame-to-frame angular distance (not cumulative)"""
        if isinstance(rotation_data, pd.DataFrame):
            quat_values = rotation_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
        else:
            # Reorder from [w, x, y, z] to [x, y, z, w] for scipy
            quat_values = rotation_data[:, [1, 2, 3, 0]]
        
        num_samples = quat_values.shape[0]
        angular_dist = np.zeros(num_samples)
        
        for i in range(num_samples - 1):
            q1 = quat_values[i]
            q2 = quat_values[i + 1]
            
            if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
               np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
                angular_dist[i] = 0
                continue
            
            try:
                r1 = R.from_quat(q1)
                r2 = R.from_quat(q2)
                
                # Calculate relative rotation
                relative_rotation = r1.inv() * r2
                
                # Angle is the norm of the rotation vector
                angle = np.linalg.norm(relative_rotation.as_rotvec())
                angular_dist[i] = angle
            except ValueError:
                angular_dist[i] = 0
        
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
        """Apply feature engineering to polars dataframe - MUST match training!"""
        # Convert to pandas for easier manipulation
        df_pd = df.to_pandas()
        
        # CRITICAL: Process each sequence separately to match training
        engineered_features = []
        
        # Get unique sequence IDs (should be just one for inference)
        sequence_ids = df_pd['sequence_id'].unique() if 'sequence_id' in df_pd.columns else [0]
        
        for seq_id in sequence_ids:
            # Extract sequence data
            if 'sequence_id' in df_pd.columns:
                seq_data = df_pd[df_pd['sequence_id'] == seq_id].copy()
            else:
                # If no sequence_id column, treat entire dataframe as one sequence
                seq_data = df_pd.copy()
                seq_data['sequence_id'] = 0
            
            # Extract base features
            acc_data = seq_data[['acc_x', 'acc_y', 'acc_z']].values
            rot_data = seq_data[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
            
            # Remove gravity from acceleration first (CRITICAL!)
            linear_accel = FeatureEngineer.remove_gravity_from_acc(acc_data, rot_data)
            seq_data[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']] = linear_accel
            
            # Calculate magnitude of linear acceleration (without gravity)
            seq_data['linear_acc_mag'] = np.linalg.norm(linear_accel, axis=1)
            
            # Calculate jerk from linear acceleration magnitude
            seq_data['linear_acc_mag_jerk'] = seq_data['linear_acc_mag'].diff().fillna(0)
            
            # Angular velocity
            angular_vel = FeatureEngineer.calculate_angular_velocity(rot_data)
            seq_data[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] = angular_vel
            
            # Angular velocity magnitude
            seq_data['angular_vel_mag'] = np.sqrt(
                angular_vel[:, 0]**2 + angular_vel[:, 1]**2 + angular_vel[:, 2]**2
            )
            
            # Angular velocity magnitude jerk
            seq_data['angular_vel_mag_jerk'] = seq_data['angular_vel_mag'].diff().fillna(0)
            
            # Gesture rhythm signature - key discriminative feature
            seq_data['gesture_rhythm_signature'] = seq_data['linear_acc_mag'].rolling(
                5, min_periods=1
            ).std() / (seq_data['linear_acc_mag'].rolling(5, min_periods=1).mean() + 1e-6)
            
            # Angular distance (frame-to-frame, not cumulative)
            seq_data['angular_distance'] = FeatureEngineer.calculate_angular_distance(rot_data)
            
            # Jerk from linear acceleration (not raw acceleration)
            jerk = FeatureEngineer.calculate_jerk(linear_accel)
            seq_data[['jerk_x', 'jerk_y', 'jerk_z']] = jerk
            
            # Keep original acceleration magnitude for compatibility
            seq_data['acc_magnitude'] = np.linalg.norm(acc_data, axis=1)
            
            # MAD features (still using raw acc for compatibility)
            acc_mad = FeatureEngineer.calculate_mad(acc_data, window_size)
            seq_data[['acc_mad_x', 'acc_mad_y', 'acc_mad_z']] = acc_mad
            
            # Rotation angle from quaternion
            seq_data['rotation_angle'] = 2 * np.arccos(np.clip(rot_data[:, 0], -1, 1))
            
            engineered_features.append(seq_data)
        
        # Concatenate all sequences
        if len(engineered_features) > 1:
            df_pd = pd.concat(engineered_features, ignore_index=True)
        else:
            df_pd = engineered_features[0]
        
        # Convert back to polars
        return pl.from_pandas(df_pd)


# Note: Quaternion helper functions are no longer needed as we use scipy.spatial.transform.Rotation


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
        self.n_imu_features = 7  # Base IMU features (now includes linear_acc instead of raw acc)
        self.n_engineered_features = 0
        
        if CONFIG['use_angular_velocity']:
            self.n_engineered_features += 5  # ang_vel(3) + mag(1) + mag_jerk(1)
        if CONFIG['use_angular_distance']:
            self.n_engineered_features += 1
        if CONFIG['use_statistical_features']:
            # Now includes: linear_acc(3) + linear_acc_mag(1) + linear_acc_mag_jerk(1) + 
            # jerk(3) + acc_mad(3) + acc_mag(1) + rot_angle(1) + rhythm_sig(1) = 14
            self.n_engineered_features += 14
            
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
        
        # Global context attention (without final softmax for masking)
        self.attention_conv = nn.Sequential(
            nn.Conv1d(1024, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, 1, kernel_size=1)
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
    
    def create_attention_mask(self, seq_len, lengths):
        """Create attention mask for padded sequences"""
        batch_size = lengths.size(0)
        max_len = seq_len
        
        # Create a tensor of sequence positions
        seq_range = torch.arange(0, max_len).to(lengths.device)
        seq_range = seq_range.unsqueeze(0).expand(batch_size, max_len)
        
        # Create mask: True for real data, False for padding
        lengths_expanded = lengths.unsqueeze(1).expand(batch_size, max_len)
        mask = seq_range < lengths_expanded
        
        return mask.float()
    
    def forward(self, x, lengths=None, sensor_mask=None):
        batch_size = x.size(0)
        seq_len = x.size(2)
        
        # Split sensor data
        imu_data = x[:, :self.n_imu_total, :]
        thm_data = x[:, self.n_imu_total:self.n_imu_total+self.n_thm_features, :]
        tof_data = x[:, self.n_imu_total+self.n_thm_features:, :]
        
        # Apply sensor masking if provided (for IMU-only sequences)
        if sensor_mask is not None:
            # sensor_mask is a boolean tensor where True = IMU-only
            # Zero out non-IMU sensors for masked samples
            thm_data = thm_data * (~sensor_mask).float().unsqueeze(1).unsqueeze(2)
            tof_data = tof_data * (~sensor_mask).float().unsqueeze(1).unsqueeze(2)
        
        # Encode each sensor type
        imu_features = self.imu_encoder(imu_data)
        thm_features = self.thm_encoder(thm_data)
        tof_features = self.tof_encoder(tof_data)
        
        # Concatenate features
        combined = torch.cat([imu_features, thm_features, tof_features], dim=1)
        
        # Apply fusion layers
        fused = self.fusion_layers(combined)
        
        # Apply attention with masking
        attention_logits = self.attention_conv(fused)  # Get attention logits
        attention_logits = attention_logits.squeeze(1)  # Remove channel dimension
        
        if lengths is not None:
            # Adjust lengths for downsampling through the network
            # We have stride=2 in encoders, so effective sequence length is reduced
            downsampled_lengths = (lengths + 1) // 2  # Account for stride=2
            # AdaptiveAvgPool1d operations don't change the sequence length conceptually
            # but we need to account for the actual output size from fusion layers
            fused_seq_len = fused.size(2)
            
            # Create attention mask for the fused sequence length
            attention_mask = self.create_attention_mask(fused_seq_len, downsampled_lengths)
            
            # Apply mask by setting padded positions to -inf before softmax
            attention_logits = attention_logits.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_logits, dim=1).unsqueeze(1)
        
        # Apply attention weights
        weighted = fused * attention_weights
        
        # Global pooling with masking
        if lengths is not None:
            # Sum only over non-padded positions
            attention_mask = attention_mask.unsqueeze(1)  # Add channel dimension
            weighted_masked = weighted * attention_mask
            global_features = weighted_masked.sum(dim=2)
            
            # Normalize by actual sequence length to get true average
            valid_lengths = attention_mask.sum(dim=2).clamp(min=1)  # Avoid division by zero
            global_features = global_features / valid_lengths
        else:
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
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
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
            
            # Handle both old (single scaler) and new (dual scaler) formats
            if 'imu_scaler' in checkpoint and 'other_scaler' in checkpoint:
                # New dual scaler format
                self.scalers.append({
                    'imu_scaler': checkpoint['imu_scaler'],
                    'other_scaler': checkpoint['other_scaler'],
                    'n_imu_features': checkpoint['n_imu_features']
                })
            else:
                # Old single scaler format for compatibility
                self.scalers.append(checkpoint['scaler'])
                
            self.label_encoders.append(label_encoder)
        
        print(f"Loaded {len(self.models)} models for ensemble")
        
        # Use first label encoder as reference
        self.label_encoder = self.label_encoders[0]
        
        # Verify label encoder loaded correctly
        print(f"Label encoder classes: {len(self.label_encoder.classes_)}")
        print(f"First few classes: {list(self.label_encoder.classes_[:3])}")
        
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
        
        # Get feature columns - MUST match training exactly!
        if self.feature_cols is None:
            # This should not happen if models are loaded correctly
            # But provide fallback that matches the training script
            # Feature columns - MUST match model's expected order:
            # 1. IMU features (linear_acc, rot, then engineered)
            # 2. Thermopile features (thm_)
            # 3. ToF features (tof_)
            
            # Collect IMU base features
            imu_base_cols = []
            imu_base_cols += [col for col in df.columns if col.startswith('linear_acc_')]
            imu_base_cols += [col for col in df.columns if col.startswith('rot_')]
            
            # Collect engineered IMU features
            engineered_cols = []
            if CONFIG['use_angular_velocity']:
                engineered_cols.extend(['ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'angular_vel_mag', 'angular_vel_mag_jerk'])
            if CONFIG['use_angular_distance']:
                engineered_cols.append('angular_distance')
            if CONFIG['use_statistical_features']:
                engineered_cols.extend(['linear_acc_mag', 'linear_acc_mag_jerk',
                                      'jerk_x', 'jerk_y', 'jerk_z', 'acc_magnitude',
                                      'acc_mad_x', 'acc_mad_y', 'acc_mad_z', 'rotation_angle',
                                      'gesture_rhythm_signature'])
            
            # Collect thermopile and ToF features
            thm_cols = [col for col in df.columns if col.startswith('thm_')]
            tof_cols = [col for col in df.columns if col.startswith('tof_')]
            
            # Build feature_cols in the correct order for the model
            self.feature_cols = imu_base_cols + engineered_cols + thm_cols + tof_cols
        
        # Extract features
        features = df[self.feature_cols].values
        
        # Handle NaN values (must match training preprocessing)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def predict_single(self, features, model_idx=0):
        """Make prediction with a single model"""
        model = self.models[model_idx]
        scaler = self.scalers[model_idx]
        
        # Scale features
        if isinstance(scaler, dict) and 'imu_scaler' in scaler:
            # New dual scaler format
            n_imu = scaler['n_imu_features']
            features_imu_scaled = scaler['imu_scaler'].transform(features[:, :n_imu])
            features_other_scaled = scaler['other_scaler'].transform(features[:, n_imu:])
            features_scaled = np.concatenate([features_imu_scaled, features_other_scaled], axis=1)
            # NO CLIPPING - removed the harmful clipping
        else:
            # Old single scaler format
            features_scaled = scaler.transform(features)
            features_scaled = np.clip(features_scaled, -10, 10)  # Keep for old models
        
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(features_scaled.T).unsqueeze(0).to(self.device)
        
        # Detect if this is an IMU-only sequence
        # Check if thermopile and ToF features are all zeros or very small
        n_imu = self.models[model_idx].n_imu_total
        n_thm = self.models[model_idx].n_thm_features
        
        # Check if non-IMU features are missing (all zeros or very small values)
        non_imu_features = features_scaled[:, n_imu:]
        is_imu_only = np.abs(non_imu_features).max() < 0.01  # Threshold for "zero"
        
        # Create sensor mask for the model
        if is_imu_only:
            sensor_mask = torch.tensor([True]).to(self.device)  # IMU-only
        else:
            sensor_mask = None  # All sensors available
        
        # Predict
        with torch.no_grad():
            outputs = model(sequence_tensor, sensor_mask=sensor_mask)
            
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