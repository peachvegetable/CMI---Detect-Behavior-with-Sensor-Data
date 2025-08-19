#!/usr/bin/env python3
"""
Improved BFRB Detection Training Script
======================================
Fixes:
1. Dynamic sequence padding per batch
2. No behavior filtering (works with test data structure)
3. Advanced IMU feature engineering
"""

import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Get the root directory of the project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm
import gc

# ===== CONFIGURATION =====
CONFIG = {
    # Training parameters
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'max_epochs': 120,
    'patience': 20,
    
    # Model parameters
    'dropout': 0.3,
    'label_smoothing': 0.15,
    'focal_gamma': 2.5,
    'mixup_alpha': 0.3,
    
    # Feature engineering
    'use_angular_velocity': True,
    'use_angular_distance': True,
    'use_statistical_features': True,
    'window_size': 10,  # for statistical features
    
    # Cross-validation
    'n_folds': 5,
    'seed': 42,
    
    # System
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
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
        """
        Calculate angular velocity from quaternion data using scipy
        rotation_data: (timesteps, 4) array of quaternions [w, x, y, z] or dataframe
        """
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
    def engineer_features(df, window_size=10):
        """Apply all feature engineering"""
        engineered_features = []
        
        for seq_id in df['sequence_id'].unique():
            seq_data = df[df['sequence_id'] == seq_id].copy()
            
            # Extract base features
            acc_data = seq_data[['acc_x', 'acc_y', 'acc_z']].values
            rot_data = seq_data[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
            
            # Remove gravity from acceleration first
            linear_accel = FeatureEngineer.remove_gravity_from_acc(acc_data, rot_data)
            seq_data[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']] = linear_accel
            
            # Calculate magnitude of linear acceleration (without gravity)
            seq_data['linear_acc_mag'] = np.linalg.norm(linear_accel, axis=1)
            
            # Calculate jerk from linear acceleration magnitude
            seq_data['linear_acc_mag_jerk'] = seq_data['linear_acc_mag'].diff().fillna(0)
            
            # Angular velocity
            angular_vel = FeatureEngineer.calculate_angular_velocity(rot_data)
            seq_data[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] = angular_vel
            
            # Angular distance (frame-to-frame, not cumulative)
            seq_data['angular_distance'] = FeatureEngineer.calculate_angular_distance(rot_data)
            
            # Jerk from linear acceleration (not raw acceleration)
            jerk = FeatureEngineer.calculate_jerk(linear_accel)
            seq_data[['jerk_x', 'jerk_y', 'jerk_z']] = jerk
            
            # Keep original acceleration magnitude for compatibility
            seq_data['acc_magnitude'] = np.linalg.norm(acc_data, axis=1)
            
            # MAD features
            acc_mad = FeatureEngineer.calculate_mad(acc_data, window_size)
            seq_data[['acc_mad_x', 'acc_mad_y', 'acc_mad_z']] = acc_mad
            
            # Rotation angle from quaternion
            seq_data['rotation_angle'] = 2 * np.arccos(np.clip(rot_data[:, 0], -1, 1))
            
            engineered_features.append(seq_data)
            
        return pd.concat(engineered_features, ignore_index=True)


# Note: Quaternion helper functions are no longer needed as we use scipy.spatial.transform.Rotation


# ===== MODEL ARCHITECTURE (same as before) =====
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
    """
    Improved model with dynamic feature count
    """
    def __init__(self, n_features, n_classes=18):
        super().__init__()
        
        # Calculate feature splits
        self.n_imu_features = 7  # Base IMU features (now includes linear_acc instead of raw acc)
        self.n_engineered_features = 0
        
        if CONFIG['use_angular_velocity']:
            self.n_engineered_features += 3
        if CONFIG['use_angular_distance']:
            self.n_engineered_features += 1
        if CONFIG['use_statistical_features']:
            # Now includes: linear_acc(3) + linear_acc_mag(1) + linear_acc_mag_jerk(1) + 
            # jerk(3) + acc_mad(3) + acc_mag(1) + rot_angle(1) = 13
            self.n_engineered_features += 13
            
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
    
    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        seq_len = x.size(2)
        
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


# ===== DATASET WITH DYNAMIC PADDING =====
class ImprovedBFRBDataset(Dataset):
    """Dataset with variable length sequences"""
    def __init__(self, sequences, binary_labels, multiclass_labels, sequence_lengths):
        self.sequences = sequences  # List of arrays with different lengths
        self.binary_labels = binary_labels
        self.multiclass_labels = multiclass_labels
        self.sequence_lengths = sequence_lengths
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return transposed sequence for Conv1d (channels, time)
        return (
            torch.FloatTensor(self.sequences[idx].T),
            self.binary_labels[idx],
            self.multiclass_labels[idx],
            self.sequence_lengths[idx]
        )

def collate_fn(batch):
    """Custom collate function for dynamic padding"""
    sequences, binary_labels, multiclass_labels, lengths = zip(*batch)
    
    # Find max sequence length in batch
    max_len = max(seq.size(1) for seq in sequences)
    
    # Manually pad sequences
    padded_sequences = []
    for seq in sequences:
        # seq is (channels, time)
        pad_len = max_len - seq.size(1)
        if pad_len > 0:
            # Pad at the end of time dimension
            padded = F.pad(seq, (0, pad_len), mode='constant', value=0)
        else:
            padded = seq
        padded_sequences.append(padded)
    
    # Stack into batch
    padded_sequences = torch.stack(padded_sequences)
    
    # Convert to tensors
    binary_labels = torch.LongTensor(binary_labels)
    multiclass_labels = torch.LongTensor(multiclass_labels)
    lengths = torch.LongTensor(lengths)
    
    return padded_sequences, binary_labels, multiclass_labels, lengths


# ===== LOSS FUNCTIONS (same as before) =====
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, n_classes, smoothing=0.0):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smoothed_label = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), confidence)
        smoothed_label += self.smoothing / self.n_classes
        return F.kl_div(F.log_softmax(pred, dim=1), smoothed_label, reduction='batchmean')


# ===== DATA AUGMENTATION =====
def mixup_data(x, y_binary, y_multi, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_binary_a, y_binary_b = y_binary, y_binary[index]
    y_multi_a, y_multi_b = y_multi, y_multi[index]
    
    return mixed_x, y_binary_a, y_binary_b, y_multi_a, y_multi_b, lam


# ===== TRAINING UTILITIES =====
def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_variable_sequences(df, feature_cols):
    """Create variable length sequences without padding"""
    sequences = []
    binary_labels = []
    multiclass_labels = []
    sequence_lengths = []
    sequence_ids = []
    
    for seq_id in tqdm(df['sequence_id'].unique(), desc="Creating sequences"):
        seq_data = df[df['sequence_id'] == seq_id]
        
        features = seq_data[feature_cols].values
        
        # Handle labels - use first value if gesture exists, otherwise default
        if 'gesture' in seq_data.columns and seq_data['gesture'].notna().any():
            binary_label = seq_data['is_bfrb'].iloc[0]
            multiclass_label = seq_data['gesture_encoded'].iloc[0]
        else:
            # For test data without gestures
            binary_label = 0
            multiclass_label = 0
        
        sequences.append(features)
        binary_labels.append(binary_label)
        multiclass_labels.append(multiclass_label)
        sequence_lengths.append(len(features))
        sequence_ids.append(seq_id)
    
    return (
        sequences,  # List of variable length arrays
        np.array(binary_labels),
        np.array(multiclass_labels),
        np.array(sequence_lengths),
        np.array(sequence_ids)
    )


def train_epoch(model, train_loader, criterion_binary, criterion_multi, optimizer, device, use_mixup=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (sequences, labels_b, labels_m, lengths) in enumerate(progress_bar):
        sequences = sequences.to(device)
        labels_b = labels_b.to(device)
        labels_m = labels_m.to(device)
        lengths = lengths.to(device)
        
        # Apply mixup augmentation
        if use_mixup and np.random.random() < 0.5:
            sequences, labels_b_a, labels_b_b, labels_m_a, labels_m_b, lam = mixup_data(
                sequences, labels_b, labels_m, CONFIG['mixup_alpha']
            )
            
            outputs = model(sequences, lengths)
            loss_binary = lam * criterion_binary(outputs['binary'], labels_b_a) + \
                        (1 - lam) * criterion_binary(outputs['binary'], labels_b_b)
            loss_multi = lam * criterion_multi(outputs['multiclass'], labels_m_a) + \
                       (1 - lam) * criterion_multi(outputs['multiclass'], labels_m_b)
        else:
            outputs = model(sequences, lengths)
            loss_binary = criterion_binary(outputs['binary'], labels_b)
            loss_multi = criterion_multi(outputs['multiclass'], labels_m)

        # Combined loss (60% binary, 40% multiclass for competition metric)
        loss = 0.6 * loss_binary + 0.4 * loss_multi
        
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Check for NaN gradients
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad = torch.zeros_like(param.grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device, criterion_binary=None, criterion_multi=None):
    """Validate the model"""
    model.eval()
    binary_preds = []
    binary_true = []
    multi_preds = []
    multi_true = []
    total_loss = 0
    
    with torch.no_grad():
        for sequences, labels_b, labels_m, lengths in tqdm(val_loader, desc='Validating'):
            sequences = sequences.to(device)
            labels_b_tensor = labels_b.to(device)
            labels_m_tensor = labels_m.to(device)
            lengths = lengths.to(device)
            
            outputs = model(sequences, lengths)
            
            # Calculate validation loss if criteria provided
            if criterion_binary is not None and criterion_multi is not None:
                loss_binary = criterion_binary(outputs['binary'], labels_b_tensor)
                loss_multi = criterion_multi(outputs['multiclass'], labels_m_tensor)
                loss = 0.6 * loss_binary + 0.4 * loss_multi
                total_loss += loss.item()
            
            binary_pred = torch.argmax(outputs['binary'], dim=1).cpu().numpy()
            multi_pred = torch.argmax(outputs['multiclass'], dim=1).cpu().numpy()
            
            binary_preds.extend(binary_pred)
            binary_true.extend(labels_b.numpy())
            multi_preds.extend(multi_pred)
            multi_true.extend(labels_m.numpy())
    
    # Calculate F1 scores
    binary_f1 = f1_score(binary_true, binary_preds, average='binary')
    multi_f1 = f1_score(multi_true, multi_preds, average='macro')
    competition_score = (binary_f1 + multi_f1) / 2
    
    result = {
        'binary_f1': binary_f1,
        'multi_f1': multi_f1,
        'competition_score': competition_score
    }
    
    if criterion_binary is not None:
        result['val_loss'] = total_loss / len(val_loader)
    
    return result


# ===== MAIN TRAINING FUNCTION =====
def main():
    print("IMPROVED BFRB DETECTION TRAINING")
    print("=" * 60)
    print(f"Configuration: {json.dumps(CONFIG, indent=2)}")
    print("=" * 60)
    
    set_seed(CONFIG['seed'])
    
    # Load data
    print("\n Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    print(f"Total samples: {len(df):,}")
    
    # NO BEHAVIOR FILTERING - work with all data
    print("\n Using all data (no behavior filtering)")
    
    # Apply feature engineering
    if CONFIG['use_angular_velocity'] or CONFIG['use_angular_distance'] or CONFIG['use_statistical_features']:
        print("\n Applying feature engineering...")
        df = FeatureEngineer.engineer_features(df, window_size=CONFIG['window_size'])
        print("Feature engineering complete!")
    
    # Create labels only for data with gestures
    if 'gesture' in df.columns:
        df['is_bfrb'] = df['gesture'].isin(BFRB_GESTURES).astype(int)
        
        # Encode multiclass labels
        label_encoder = LabelEncoder()
        # Handle NaN gestures by filling with a placeholder
        df['gesture_filled'] = df['gesture'].fillna('no_gesture')
        df['gesture_encoded'] = label_encoder.fit_transform(df['gesture_filled'])
        
        print(f"\nClasses: {len(label_encoder.classes_)}")
        print(f"BFRB samples: {df['is_bfrb'].sum():,}")
        print(f"Non-BFRB samples: {(1-df['is_bfrb']).sum():,}")
    else:
        # For test data without gestures
        df['is_bfrb'] = 0
        df['gesture_encoded'] = 0
        label_encoder = None
    
    # Feature columns (including engineered features)
    # Note: We now use linear_acc instead of raw acc for better feature representation
    base_feature_cols = [col for col in df.columns if col.startswith(('linear_acc_', 'rot_', 'thm_', 'tof_'))]
    # Keep acc_ columns for compatibility but they shouldn't be primary features
    base_feature_cols += [col for col in df.columns if col.startswith('acc_') and col not in base_feature_cols]
    engineered_cols = []
    
    if CONFIG['use_angular_velocity']:
        engineered_cols.extend(['ang_vel_x', 'ang_vel_y', 'ang_vel_z'])
    if CONFIG['use_angular_distance']:
        engineered_cols.append('angular_distance')
    if CONFIG['use_statistical_features']:
        engineered_cols.extend(['linear_acc_mag', 'linear_acc_mag_jerk',
                              'jerk_x', 'jerk_y', 'jerk_z', 'acc_magnitude',
                              'acc_mad_x', 'acc_mad_y', 'acc_mad_z', 'rotation_angle'])
    
    feature_cols = base_feature_cols + engineered_cols
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Base features: {len(base_feature_cols)}")
    print(f"Engineered features: {len(engineered_cols)}")
    
    # Create variable length sequences
    print("\n Creating variable length sequences...")
    X, y_binary, y_multi, seq_lengths, seq_ids = create_variable_sequences(
        df, feature_cols
    )
    
    print(f"Total sequences: {len(X):,}")
    print(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")
    
    # Only do cross-validation if we have real labels
    if 'gesture' in df.columns:
        # Cross-validation
        print("\n Starting cross-validation...")
        skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_multi)):
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{CONFIG['n_folds']}")
            print(f"{'='*60}")
            
            # Split data
            X_train = [X[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_binary_train, y_binary_val = y_binary[train_idx], y_binary[val_idx]
            y_multi_train, y_multi_val = y_multi[train_idx], y_multi[val_idx]
            seq_lengths_train = seq_lengths[train_idx]
            seq_lengths_val = seq_lengths[val_idx]
            
            # Normalize each sequence independently
            # Use StandardScaler for better compatibility with test distribution
            scaler = StandardScaler()
            
            # Fit scaler on concatenated training data
            X_train_concat = np.vstack(X_train)
            X_train_concat = np.nan_to_num(X_train_concat, nan=0.0, posinf=0.0, neginf=0.0)
            scaler.fit(X_train_concat)
            
            # Apply scaling to each sequence
            X_train_scaled = []
            for seq in X_train:
                seq_clean = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
                seq_scaled = scaler.transform(seq_clean)
                seq_scaled = np.clip(seq_scaled, -10, 10)
                X_train_scaled.append(seq_scaled)
            
            X_val_scaled = []
            for seq in X_val:
                seq_clean = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
                seq_scaled = scaler.transform(seq_clean)
                seq_scaled = np.clip(seq_scaled, -10, 10)
                X_val_scaled.append(seq_scaled)
            
            # Create datasets
            train_dataset = ImprovedBFRBDataset(X_train_scaled, y_binary_train, y_multi_train, seq_lengths_train)
            val_dataset = ImprovedBFRBDataset(X_val_scaled, y_binary_val, y_multi_val, seq_lengths_val)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                num_workers=CONFIG['num_workers'],
                pin_memory=True,
                collate_fn=collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=False,
                num_workers=CONFIG['num_workers'],
                pin_memory=True,
                collate_fn=collate_fn
            )
            
            # Initialize model
            n_classes = len(label_encoder.classes_) if label_encoder else 18
            model = ImprovedBFRBModel(n_features=len(feature_cols), n_classes=n_classes).to(CONFIG['device'])
            
            # Loss functions
            binary_weights = compute_class_weight('balanced', classes=np.unique(y_binary_train), y=y_binary_train)
            binary_weights = torch.FloatTensor(binary_weights).to(CONFIG['device'])
            
            multi_weights = compute_class_weight('balanced', classes=np.unique(y_multi_train), y=y_multi_train)
            multi_weights = torch.FloatTensor(multi_weights).to(CONFIG['device'])
            
            criterion_binary = FocalLoss(gamma=CONFIG['focal_gamma'], alpha=binary_weights)
            criterion_multi = LabelSmoothingLoss(n_classes=n_classes, smoothing=CONFIG['label_smoothing'])
            
            # Optimizer and scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=CONFIG['learning_rate'],
                weight_decay=CONFIG['weight_decay']
            )
            
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
            
            # Training loop
            best_score = 0
            patience_counter = 0
            best_epoch = 0
            
            train_losses = []
            val_scores = []
            val_losses = []
            
            for epoch in range(CONFIG['max_epochs']):
                print(f"\nEpoch {epoch + 1}/{CONFIG['max_epochs']}")
                
                # Train
                train_loss = train_epoch(
                    model, train_loader, criterion_binary, criterion_multi, 
                    optimizer, CONFIG['device'], use_mixup=True
                )
                train_losses.append(train_loss)
                
                # Validate
                val_metrics = validate(model, val_loader, CONFIG['device'], criterion_binary, criterion_multi)
                val_scores.append(val_metrics['competition_score'])
                val_losses.append(val_metrics['val_loss'])
                
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"Val Binary F1: {val_metrics['binary_f1']:.4f}")
                print(f"Val Multi F1: {val_metrics['multi_f1']:.4f}")
                print(f"Val Score: {val_metrics['competition_score']:.4f}")
                
                # Learning rate scheduling
                scheduler.step()
                
                # Save best model
                if val_metrics['competition_score'] > best_score:
                    best_score = val_metrics['competition_score']
                    best_epoch = epoch
                    patience_counter = 0
                    
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'scaler': scaler,
                        'label_encoder': label_encoder,
                        'score': best_score,
                        'epoch': epoch,
                        'config': CONFIG,
                        'feature_cols': feature_cols,
                        'n_features': len(feature_cols)
                    }, os.path.join(PROJECT_ROOT, f'improved_model_fold_{fold}.pth'))
                    
                    print(f"Saved best model (score: {best_score:.4f})")
                else:
                    patience_counter += 1
                    print(f"Patience: {patience_counter}/{CONFIG['patience']}")
                    
                    # Early stopping
                    if patience_counter >= CONFIG['patience']:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        print(f"Best score: {best_score:.4f} at epoch {best_epoch + 1}")
                        break
            
            cv_scores.append(best_score)
            print(f"\nFold {fold + 1} best score: {best_score:.4f}")
            
            # Clean up
            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
        
        # Final results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"CV Scores: {cv_scores}")
        print(f"Mean Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"Best Score: {max(cv_scores):.4f}")
        
        # Save results
        results = {
            'cv_scores': cv_scores,
            'mean_score': float(np.mean(cv_scores)),
            'std_score': float(np.std(cv_scores)),
            'best_score': float(max(cv_scores)),
            'config': CONFIG
        }
        
        with open(os.path.join(PROJECT_ROOT, 'improved_training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    print("\n Training complete!")
    print("Models saved as: improved_model_fold_*.pth")


if __name__ == "__main__":
    main()