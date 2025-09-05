#!/usr/bin/env python3
"""
PyTorch 3-Model Ensemble for CMI Competition
Consistent with original TensorFlow notebook feature engineering
Combines three architectures:
1. GatedTwoBranchModel - Adaptive sensor fusion with gating
2. TwoBranchMixedModel - LSTM/GRU/Dense parallel paths  
3. CMIBertModel - BERT transformer for multi-modal fusion

Can be used for both training and Kaggle submission.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler

from scipy.spatial.transform import Rotation as R
from scipy.signal import firwin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from transformers import BertConfig, BertModel
from tqdm import tqdm
import gc
import math

# Import polars for inference if available
try:
    import polars as pl
except ImportError:
    pl = None

# ==================== Configuration ====================

# Training/Inference mode switch
TRAIN = True  # Set to False for Kaggle submission

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available() and TRAIN:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif TRAIN:
    print("Using CPU")

# Configuration
CONFIG = {
    'batch_size': 64,  # Match original notebook
    'pad_percentile': 95,
    'lr_init': 4e-4,  # Match original notebook (LR_INIT)
    'weight_decay': 5e-4,  # Reduced from 3e-3 for better convergence (WD)
    'mixup_alpha': 0.4,  # Match original notebook
    'epochs': 360,  # Match original notebook for gated model
    'patience': 50,  # Match original notebook
    'n_splits': 10,  # Match original notebook (10-fold as shown)
    'masking_prob': 0.25,  # Match original notebook
    'gate_loss_weight': 0.2,  # Gate supervision weight
    'label_smoothing': 0.1,  # Original notebook uses 0.1 for gated model
    'tta_steps': 10,  # Test-time augmentation steps
    'tta_noise_stddev': 0.01,  # Noise std for TTA
    'ensemble_weights': [0.53, 0.18, 0.29],  # Fixed order: [gated=0.53, mixed=0.18, bert=0.29]
    'device': device,
    'seed': 42,
    'use_feature_engineering': True  # Match original notebook
}

# Set seeds for reproducibility
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG['seed'])

# Data paths and imports based on mode
if TRAIN:
    from competition_metric import CompetitionMetric
    RAW_DIR = Path("data")
    EXPORT_DIR = Path("./ensemble_model")
    EXPORT_DIR.mkdir(exist_ok=True)
    from competition_metric import CompetitionMetric
    print(f"\nTRAIN MODE - Starting 3-Model Ensemble Training")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Device: {CONFIG['device']}")
else:
    # For Kaggle submission
    RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
    PRETRAINED_DIR = Path("/kaggle/input/ensemble/pytorch/default/4")
    print("▶ INFERENCE MODE - Loading pretrained 3-model ensemble for submission")

# Removed unused normalization constants - data is normalized via StandardScaler

# ==================== Physics-based Feature Engineering ====================

def remove_gravity_from_acc(acc_data, rot_data):
    """Remove gravity from accelerometer data using quaternion rotation"""
    acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    
    for i in range(len(acc_values)):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except (ValueError, IndexError):
            linear_accel[i, :] = acc_values[i, :]
    
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    """Calculate angular velocity from quaternion data"""
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_vel = np.zeros((len(quat_values), 3))
    
    for i in range(len(quat_values) - 1):
        q_t, q_t_plus_dt = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q_t)) or np.all(np.isnan(q_t_plus_dt)):
            continue
        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except (ValueError, IndexError):
            pass
    
    return angular_vel

def calculate_angular_distance(rot_data):
    """Calculate angular distance between consecutive quaternions"""
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_dist = np.zeros(len(quat_values))
    
    for i in range(len(quat_values) - 1):
        q1, q2 = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q1)) or np.all(np.isnan(q2)):
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angular_dist[i] = np.linalg.norm(relative_rotation.as_rotvec())
        except (ValueError, IndexError):
            pass
    
    return angular_dist

def engineer_features(df_seq):
    """Apply feature engineering consistent with original notebook"""
    # Remove gravity from acceleration
    if all(col in df_seq.columns for col in ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']):
        linear_accel = remove_gravity_from_acc(df_seq, df_seq)
        df_seq['linear_acc_x'] = linear_accel[:, 0]
        df_seq['linear_acc_y'] = linear_accel[:, 1]
        df_seq['linear_acc_z'] = linear_accel[:, 2]
        
        # Calculate angular velocity
        angular_vel = calculate_angular_velocity_from_quat(df_seq)
        df_seq['angular_vel_x'] = angular_vel[:, 0]
        df_seq['angular_vel_y'] = angular_vel[:, 1]
        df_seq['angular_vel_z'] = angular_vel[:, 2]
        
        # Calculate angular distance
        angular_dist = calculate_angular_distance(df_seq)
        df_seq['angular_distance'] = angular_dist
        
        # Additional engineered features from original notebook
        # Acceleration magnitude
        df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
        df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2)
        
        # Jerk (derivative of acceleration)
        for axis in ['x', 'y', 'z']:
            acc_col = f'acc_{axis}'
            jerk_col = f'acc_jerk_{axis}'
            df_seq[jerk_col] = df_seq[acc_col].diff().fillna(0)
        
        # Rotation angle from quaternion
        df_seq['rot_angle'] = 2 * np.arccos(np.clip(df_seq['rot_w'].values, -1, 1))
    
    return df_seq

def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    """Pad sequences to the same length"""
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            seq = seq[:maxlen] if truncating == 'post' else seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            pad_array = np.full((pad_len, seq.shape[1]), value)
            seq = np.concatenate([seq, pad_array]) if padding == 'post' else \
                  np.concatenate([pad_array, seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)

# ==================== Model Components ====================

class ImuFeatureExtractor(nn.Module):
    """IMU Feature Extractor for preprocessing accelerometer and gyroscope data"""
    def __init__(self, fs=100., add_quaternion=False):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion
        
        k = 15
        # Note: self.lpf is unused but kept for backward compatibility
        # self.lpf = nn.Conv1d(6, 6, kernel_size=k, padding=k//2, groups=6, bias=False)
        # nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))
        
        self.lpf_acc = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        nn.init.kaiming_uniform_(self.lpf_acc.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lpf_gyro.weight, a=math.sqrt(5))
    
    def forward(self, imu):
        # Expecting 6 channels: 3 acc + 3 gyro/angular_vel
        if imu.shape[1] < 6:
            # Pad with zeros if we have fewer channels
            padding = torch.zeros(imu.shape[0], 6 - imu.shape[1], imu.shape[2]).to(imu.device)
            imu = torch.cat([imu, padding], dim=1)
        
        acc = imu[:, 0:3, :]
        gyro = imu[:, 3:6, :]
        
        # 1) Magnitude
        acc_mag = torch.norm(acc, dim=1, keepdim=True)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)
        
        # 2) Jerk (derivative)
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1, 0))
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1, 0))
        
        # 3) Energy (power)
        acc_pow = acc ** 2
        gyro_pow = gyro ** 2
        
        # 4) Low-pass and high-pass filtering
        acc_lpf = self.lpf_acc(acc)
        acc_hpf = acc - acc_lpf
        gyro_lpf = self.lpf_gyro(gyro)
        gyro_hpf = gyro - gyro_lpf
        
        features = [
            acc, gyro,
            acc_mag, gyro_mag,
            jerk, gyro_delta,
            acc_pow, gyro_pow,
            acc_lpf, acc_hpf,
            gyro_lpf, gyro_hpf,
        ]
        return torch.cat(features, dim=1)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=8):
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

class ResidualSEBlock(nn.Module):
    """Residual block with SE attention - from dual_gate"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(2)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        return out

# Keep for compatibility with existing code
class ResidualSECNNBlock(ResidualSEBlock):
    """Alias for compatibility"""
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
        # pool_size parameter is kept for signature compatibility but not used
        super().__init__(in_channels, out_channels, kernel_size, stride=1, dropout=dropout)

class AttentionLayer(nn.Module):
    """Attention mechanism for sequence aggregation - from dual_gate"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (x * weights).sum(dim=1)  # (batch, hidden_dim)
        return context

# ==================== Model A: Gated Two-Branch Model (from dual_gate) ====================

class GatedTwoBranchModel(nn.Module):
    """Two-branch model with gating mechanism - processes IMU and TOF+THM branches separately"""
    def __init__(self, pad_len=None, imu_dim=None, tof_thm_dim=None, n_classes=None, 
                 imu_dim_raw=None, tof_dim=None, feature_engineering=False, **kwargs):
        super().__init__()
        
        # Handle different call signatures for compatibility
        if imu_dim is not None:
            # New unified signature
            self.imu_dim = imu_dim
            self.tof_thm_dim = tof_thm_dim if tof_thm_dim is not None else tof_dim
        elif pad_len is not None and imu_dim_raw is not None:
            # Old signature from three_model_ensemble
            self.imu_dim = imu_dim_raw
            self.tof_thm_dim = tof_dim
        else:
            # Fallback from kwargs
            self.imu_dim = kwargs.get('imu_dim', 6)
            self.tof_thm_dim = kwargs.get('tof_thm_dim', kwargs.get('tof_dim', 6))
        
        self.n_classes = n_classes if n_classes is not None else kwargs.get('n_classes', 5)
        
        # IMU branch (deep)
        self.imu_branch = nn.Sequential(
            ResidualSEBlock(self.imu_dim, 64, kernel_size=3, dropout=0.1),
            ResidualSEBlock(64, 128, kernel_size=5, dropout=0.1)
        )
        
        # TOF/THM branch (lighter) - uses combined tof_thm_dim
        self.tof_branch = nn.Sequential(
            nn.Conv1d(self.tof_thm_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # Gate network - takes raw TOF/THM input, outputs logits (no sigmoid)
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.tof_thm_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)  # Output logits, no sigmoid
        )
        
        # Fusion layers
        fusion_dim = 128 + 128  # IMU + TOF branches
        
        # BiLSTM branch
        self.lstm = nn.LSTM(fusion_dim, 128, bidirectional=True, batch_first=True)
        
        # BiGRU branch
        self.gru = nn.GRU(fusion_dim, 128, bidirectional=True, batch_first=True)
        
        # Noise branch
        self.noise_layer = nn.Sequential(
            nn.Linear(fusion_dim, 16),
            nn.ELU()
        )
        
        # Attention aggregation
        combined_dim = 256 + 256 + 16  # BiLSTM + BiGRU + Noise
        self.attention = AttentionLayer(combined_dim)
        
        # Classification head (matches TF notebook: 512->256->128)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, self.n_classes)
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
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Handle both input formats
        if len(x.shape) == 3:
            # Input is (batch, seq_len, features) - transpose to (batch, features, seq_len)
            x = x.transpose(1, 2)
        
        # Split input into IMU and TOF/THM
        imu_data = x[:, :self.imu_dim, :]
        
        # Handle case where TOF/THM might be missing
        if x.shape[1] > self.imu_dim:
            tof_thm_data = x[:, self.imu_dim:self.imu_dim+self.tof_thm_dim, :]
        else:
            # IMU-only case: create zeros with correct channel count
            tof_thm_data = torch.zeros(batch_size, self.tof_thm_dim, x.shape[2], 
                                       device=x.device, dtype=x.dtype)
        
        # Process IMU branch
        imu_features = self.imu_branch(imu_data)
        
        # Process TOF/THM branch with gating
        tof_features_base = self.tof_branch(tof_thm_data)
        # Feed gate from raw TOF/THM data (matches original TF implementation)
        gate_logits = self.gate_network(tof_thm_data)  # (batch, 1)
        gate_probs = torch.sigmoid(gate_logits)  # Apply sigmoid for gating
        gate_expanded = gate_probs.unsqueeze(1)  # (batch, 1, 1)
        tof_features = tof_features_base * gate_expanded
        
        # Concatenate branches
        combined = torch.cat([imu_features, tof_features], dim=1)
        
        # Permute for RNN layers (batch, channels, seq_len) -> (batch, seq_len, channels)
        combined = combined.permute(0, 2, 1)
        
        # Apply RNN branches
        lstm_out, _ = self.lstm(combined)
        gru_out, _ = self.gru(combined)
        
        # Apply noise branch with Gaussian noise during training
        if self.training:
            noise = torch.randn_like(combined) * 0.05  # Reduced from 0.09 for stability
            noise_out = self.noise_layer(combined + noise)
        else:
            noise_out = self.noise_layer(combined)
        
        # Concatenate all branches
        merged = torch.cat([lstm_out, gru_out, noise_out], dim=-1)
        
        # Apply dropout (matching original TF model)
        merged = F.dropout(merged, p=0.4, training=self.training)
        
        # Apply attention and aggregate
        context = self.attention(merged)
        
        # Classification
        output = self.classifier(context)
        
        # Return classification output and gate logits (not probabilities)
        return output, gate_logits.squeeze(-1)

# ==================== Model B: Two-Branch Mixed Model ====================

class TwoBranchMixedModel(nn.Module):
    """Two-branch model with mixed LSTM/GRU/Dense paths"""
    def __init__(self, pad_len=None, imu_dim=None, tof_dim=None, n_classes=None, **kwargs):
        super().__init__()
        self.imu_dim = imu_dim if imu_dim is not None else 6
        self.tof_dim = tof_dim if tof_dim is not None else 6
        self.n_classes = n_classes if n_classes is not None else 5
        
        # IMU deep branch
        self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3, dropout=0.1)
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5, dropout=0.1)
        
        # TOF/THM lighter branch
        self.tof_conv1 = nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False)
        self.tof_bn1 = nn.BatchNorm1d(64)
        self.tof_pool1 = nn.MaxPool1d(2)
        self.tof_drop1 = nn.Dropout(0.2)
        
        self.tof_conv2 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.tof_bn2 = nn.BatchNorm1d(128)
        self.tof_pool2 = nn.MaxPool1d(2)
        self.tof_drop2 = nn.Dropout(0.2)
        
        # Three parallel paths after merge
        self.bilstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.bigru = nn.GRU(256, 128, bidirectional=True, batch_first=True)
        
        self.noise_std = 0.05  # Reduced from 0.09 for stability
        self.dense_path = nn.Sequential(
            nn.Linear(256, 16),
            nn.ELU()
        )
        
        # Attention for aggregating sequences
        self.attention = AttentionLayer(256 + 256 + 16)
        
        # Output layers
        self.dropout_main = nn.Dropout(0.4)
        
        self.dense1 = nn.Linear(256 + 256 + 16, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        
        self.dense2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        
        self.classifier = nn.Linear(128, n_classes)
    
    def forward(self, x):
        # Handle both input formats
        if len(x.shape) == 3:
            # Split input from (batch, seq_len, features)
            imu = x[:, :, :self.imu_dim].transpose(1, 2)
            tof = x[:, :, self.imu_dim:self.imu_dim+self.tof_dim].transpose(1, 2)
        else:
            # Already in (batch, features, seq_len) format
            imu = x[:, :self.imu_dim, :]
            tof = x[:, self.imu_dim:self.imu_dim+self.tof_dim, :]
        
        # IMU branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        
        # TOF branch
        x2 = self.tof_drop1(self.tof_pool1(F.relu(self.tof_bn1(self.tof_conv1(tof)))))
        x2 = self.tof_drop2(self.tof_pool2(F.relu(self.tof_bn2(self.tof_conv2(x2)))))
        
        # Merge branches
        merged = torch.cat([x1, x2], dim=1).transpose(1, 2)
        
        # Three parallel paths
        xa, _ = self.bilstm(merged)
        xb, _ = self.bigru(merged)
        
        if self.training:
            noise = torch.randn_like(merged) * self.noise_std
            xc = merged + noise
        else:
            xc = merged
        xc = self.dense_path(xc)
        
        # Concatenate all paths
        x = torch.cat([xa, xb, xc], dim=-1)
        x = self.dropout_main(x)
        
        # Attention aggregation
        x = self.attention(x)
        
        # Dense layers
        x = self.drop1(F.relu(self.bn1(self.dense1(x))))
        x = self.drop2(F.relu(self.bn2(self.dense2(x))))
        
        # Classification
        output = self.classifier(x)
        
        return output

# ==================== Model C: BERT Model ====================

class CMIBertModel(nn.Module):
    """BERT-based model for CMI competition with attention masking for padded sequences"""
    def __init__(self, imu_dim=6, tof_dim=5, thm_dim=1, n_classes=5, **kwargs):
        super().__init__()
        
        # Feature dimensions with safety guards
        self.imu_dim = max(1, imu_dim)
        self.tof_dim = max(1, tof_dim)
        self.thm_dim = max(1, thm_dim)  # Ensure thm_dim is at least 1 to avoid Conv1d(0, ...)
        
        # Feature extractors
        self.imu_conv = nn.Conv1d(imu_dim, kwargs.get("imu_conv_dim", 128), 
                                  kernel_size=3, padding=1)
        self.tof_conv = nn.Conv1d(tof_dim, kwargs.get("tof_conv_dim", 64), 
                                  kernel_size=3, padding=1)
        self.thm_conv = nn.Conv1d(thm_dim, kwargs.get("thm_conv_dim", 32), 
                                  kernel_size=3, padding=1)
        
        # Batch normalization
        self.imu_bn = nn.BatchNorm1d(kwargs.get("imu_conv_dim", 128))
        self.tof_bn = nn.BatchNorm1d(kwargs.get("tof_conv_dim", 64))
        self.thm_bn = nn.BatchNorm1d(kwargs.get("thm_conv_dim", 32))
        
        # BERT configuration
        self.bert_hidden_size = (kwargs.get("imu_conv_dim", 128) + 
                                 kwargs.get("tof_conv_dim", 64) + 
                                 kwargs.get("thm_conv_dim", 32))
        
        self.bert_config = BertConfig(
            hidden_size=self.bert_hidden_size,
            num_hidden_layers=kwargs.get("bert_layers", 4),  # Reduced for speed
            num_attention_heads=kwargs.get("bert_heads", 8),
            intermediate_size=self.bert_hidden_size * 4,
            hidden_dropout_prob=kwargs.get("dropout", 0.1),
            attention_probs_dropout_prob=kwargs.get("dropout", 0.1),
            max_position_embeddings=1024,
        )
        
        self.bert = BertModel(self.bert_config)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.bert_hidden_size))
        
        # Classifier head
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.1))
        self.classifier = nn.Linear(self.bert_hidden_size, n_classes)
    
    def forward(self, x, pad_mask=None):
        # For unified interface, split combined input
        imu = x[:, :, :self.imu_dim].transpose(1, 2)
        thm = x[:, :, self.imu_dim:self.imu_dim+self.thm_dim].transpose(1, 2) if x.shape[-1] > self.imu_dim else torch.zeros(x.shape[0], self.thm_dim, x.shape[1], device=x.device)
        tof = x[:, :, self.imu_dim+self.thm_dim:].transpose(1, 2) if x.shape[-1] > self.imu_dim+self.thm_dim else torch.zeros(x.shape[0], self.tof_dim, x.shape[1], device=x.device)
        
        # Feature extraction
        imu_feat = F.relu(self.imu_bn(self.imu_conv(imu)))
        tof_feat = F.relu(self.tof_bn(self.tof_conv(tof)))
        thm_feat = F.relu(self.thm_bn(self.thm_conv(thm)))
        
        # Create attention mask based on padding information
        if pad_mask is not None:
            # Use provided padding mask (True = padded, False = valid)
            timestep_mask = ~pad_mask  # Invert: attention mask needs True for valid positions
        else:
            # Fallback to feature-based detection if no mask provided
            feature_sum = (imu_feat.abs().sum(1) + thm_feat.abs().sum(1) + tof_feat.abs().sum(1))
            timestep_mask = feature_sum > 1e-8
        
        # Concatenate features
        bert_input = torch.cat([imu_feat, thm_feat, tof_feat], dim=1).permute(0, 2, 1)
        
        # Add CLS token
        cls_token = self.cls_token.expand(bert_input.size(0), -1, -1)
        bert_input = torch.cat([cls_token, bert_input], dim=1)
        
        # Create attention mask with CLS token always attended
        batch_size = timestep_mask.size(0)
        cls_mask = torch.ones(batch_size, 1, device=timestep_mask.device, dtype=timestep_mask.dtype)
        attention_mask = torch.cat([cls_mask, timestep_mask], dim=1)
        # Cast to long and ensure correct device
        attention_mask = attention_mask.to(dtype=torch.long, device=bert_input.device)
        
        # BERT forward pass with attention mask
        outputs = self.bert(inputs_embeds=bert_input, attention_mask=attention_mask)
        
        # Use CLS token for classification
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Classification
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits

# ==================== Dataset ====================

class CMIDataset(Dataset):
    def __init__(self, X, y=None, pad_mask=None, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = y
        self.pad_mask = torch.BoolTensor(pad_mask) if pad_mask is not None else None
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        if self.transform:
            x = self.transform(x)
            
        if self.y is not None:
            if self.pad_mask is not None:
                return x, self.y[idx], self.pad_mask[idx]
            return x, self.y[idx]
        
        if self.pad_mask is not None:
            return x, self.pad_mask[idx]
        return x

# ==================== Training Functions ====================

def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def train_epoch(model, loader, criterion, optimizer, scaler, device, is_gated=False, masking_prob=0.0, mixup_alpha=0.0, imu_dim=None, thm_dim=0):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch in tqdm(loader, desc="Training"):
        pad_mask_batch = None
        if len(batch) == 3:
            x, y, pad_mask_batch = batch
            x, y = x.to(device), y.to(device)
            pad_mask_batch = pad_mask_batch.to(device)
        elif len(batch) == 2:
            x, y = batch
            x, y = x.to(device), y.to(device)
        else:
            x = batch
            x = x.to(device)
            y = None
        
        # Apply mixup augmentation
        if mixup_alpha > 0 and y is not None:
            x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
        else:
            y_a = y_b = y
            lam = 1.0
        
        # Apply per-sample sensor masking (TOF/THM sensors) for gated model
        sample_mask = None
        if is_gated and masking_prob > 0:
            batch_size = x.size(0)
            # Per-sample mask: True = mask this sample's non-IMU sensors
            sample_mask = torch.rand(batch_size, device=x.device) < masking_prob
            if sample_mask.any():
                non_imu_start = imu_dim  # Start of non-IMU features
                if non_imu_start < x.shape[-1]:
                    x = x.clone()
                    # Mask non-IMU channels only for selected samples
                    x[sample_mask, :, non_imu_start:] = 0
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            if is_gated:
                output, gate_logits = model(x)
                # Main classification loss with mixup
                if lam < 1.0 and y_a is not None:
                    ce_loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
                else:
                    ce_loss = criterion(output, y_a) if y_a is not None else 0
                
                # Per-sample gate supervision
                if sample_mask is not None:
                    # Gate target: 1 when sensors present (not masked), 0 when masked
                    gate_target = (~sample_mask).float()
                else:
                    gate_target = torch.ones(gate_logits.size(0), device=gate_logits.device)
                gate_loss = F.binary_cross_entropy_with_logits(gate_logits.squeeze(-1), gate_target)
                
                # Combined loss with gate supervision
                loss = ce_loss + CONFIG['gate_loss_weight'] * gate_loss
            elif isinstance(model, CMIBertModel):
                # Pass padding mask to BERT model
                output = model(x, pad_mask=pad_mask_batch)
                # Compute loss (supports mixup if active)
                if lam < 1.0 and y_a is not None:
                    loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
                else:
                    loss = criterion(output, y_a) if y_a is not None else 0
            else:
                output = model(x)
                # Mixup loss
                if lam < 1.0 and y_a is not None:
                    loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
                else:
                    loss = criterion(output, y_a) if y_a is not None else 0
        
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        batch_count += 1
    
    # Return average loss, handle case where no batches were processed
    return total_loss / max(batch_count, 1)

def validate_epoch(model, loader, criterion, device, is_gated=False, label_encoder=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            pad_mask_batch = None
            if len(batch) == 3:
                x, y, pad_mask_batch = batch
                x, y = x.to(device), y.to(device)
                pad_mask_batch = pad_mask_batch.to(device)
            elif len(batch) == 2:
                x, y = batch
                x, y = x.to(device), y.to(device)
            else:
                x = batch
                x = x.to(device)
                y = None
            
            if is_gated:
                output, _ = model(x)
            elif isinstance(model, CMIBertModel):
                output = model(x, pad_mask=pad_mask_batch)
            else:
                output = model(x)
            
            if y is not None:
                loss = criterion(output, y)
                total_loss += loss.item()
                
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
    
    if all_labels:
        # Use CompetitionMetric for hierarchical F1 score
        if label_encoder is not None:
            y_true = label_encoder.inverse_transform(all_labels)
            y_pred = label_encoder.inverse_transform(all_preds)
            
            metric = CompetitionMetric()
            score = metric.calculate_hierarchical_f1(
                pd.DataFrame({'gesture': y_true}),
                pd.DataFrame({'gesture': y_pred})
            )
        else:
            score = f1_score(all_labels, all_preds, average='macro')
    else:
        score = 0
    
    return total_loss / len(loader) if len(loader) > 0 else 0, score

# ==================== Main Training Function ====================

def train_models():
    """Train all three models with feature engineering matching original notebook"""
    print("\n" + "="*60)
    print("Starting 3-Model Ensemble Training")
    print("Feature engineering:", "ENABLED" if CONFIG['use_feature_engineering'] else "DISABLED")
    print("="*60)
    
    # Load training data
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    # test_df = pd.read_csv(RAW_DIR / "test.csv")  # Not needed during training
    
    # Prepare label encoder
    label_encoder = LabelEncoder()
    train_df['gesture_encoded'] = label_encoder.fit_transform(train_df['gesture'])
    gesture_classes = label_encoder.classes_
    n_classes = len(gesture_classes)
    
    print(f"Found {n_classes} gesture classes: {gesture_classes}")
    
    # Group by sequence and apply feature engineering if enabled
    X_list = []
    y_list = []
    subject_list = []
    feature_cols_global = []  # Collect all unique feature columns across sequences
    
    print("Processing sequences...")
    for (subject, gesture, _), seq_group in tqdm(train_df.groupby(['subject', 'gesture', 'sequence_id'])):
        # Apply feature engineering to the sequence
        if CONFIG['use_feature_engineering']:
            seq_group = engineer_features(seq_group.copy())
            
            # Define feature columns consistent with original notebook
            feature_cols = []
            
            # Linear acceleration (gravity removed)
            if all(col in seq_group.columns for col in ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']):
                feature_cols.extend(['linear_acc_x', 'linear_acc_y', 'linear_acc_z'])
            else:
                feature_cols.extend(['acc_x', 'acc_y', 'acc_z'])
            
            # Angular velocity
            if all(col in seq_group.columns for col in ['angular_vel_x', 'angular_vel_y', 'angular_vel_z']):
                feature_cols.extend(['angular_vel_x', 'angular_vel_y', 'angular_vel_z'])
            
            # Rotation quaternions
            rot_cols = [col for col in seq_group.columns if col.startswith('rot_') and col in ['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            feature_cols.extend(rot_cols)
            
            # Angular distance
            if 'angular_distance' in seq_group.columns:
                feature_cols.append('angular_distance')
            
            # Additional engineered features
            eng_features = ['acc_mag', 'linear_acc_mag', 'rot_angle']
            jerk_features = [f'acc_jerk_{axis}' for axis in ['x', 'y', 'z']]
            for feat in eng_features + jerk_features:
                if feat in seq_group.columns:
                    feature_cols.append(feat)
            
            # Thermal columns
            thm_cols = [col for col in seq_group.columns if col.startswith('thm_')]
            feature_cols.extend(thm_cols)
            
            # TOF columns  
            tof_cols = [col for col in seq_group.columns if col.startswith('tof_')]
            feature_cols.extend(tof_cols)
        else:
            # Use raw sensor columns
            sensor_prefixes = ['acc_', 'rot_', 'thm_', 'tof_']
            feature_cols = [col for col in seq_group.columns 
                          if any(col.startswith(prefix) for prefix in sensor_prefixes)]
        
        # Add new feature columns to global list (preserving order)
        for col in feature_cols:
            if col not in feature_cols_global:
                feature_cols_global.append(col)
        
        # Extract sequence data with forward/backward fill like original
        # Forward fill then backward fill for missing values
        seq_filled = seq_group[feature_cols].ffill().bfill().fillna(0)
        seq_data = seq_filled.values.astype(np.float32)
        
        if len(seq_data) > 0:
            X_list.append(seq_data)
            y_list.append(gesture)
            subject_list.append(subject)
    
    print(f"Processed {len(X_list)} sequences")
    
    # Pad sequences
    pad_percentile = CONFIG['pad_percentile']
    sequence_lengths = [len(seq) for seq in X_list]
    pad_len = int(np.percentile(sequence_lengths, pad_percentile))
    print(f"Padding sequences to length {pad_len} (p{pad_percentile})")
    
    X = pad_sequences_torch(X_list, pad_len)
    y = label_encoder.transform(y_list)
    subjects = np.array(subject_list)
    
    # Create padding mask BEFORE normalization (True where padded)
    pad_mask = np.zeros((len(X_list), pad_len), dtype=bool)
    for i, orig_len in enumerate(sequence_lengths):
        if orig_len < pad_len:
            pad_mask[i, orig_len:] = True
    
    # Normalize features - fit only on non-padded data
    scaler = StandardScaler()
    X_flat = X.reshape(-1, X.shape[-1])
    mask_flat = pad_mask.reshape(-1)  # True = padded
    # CRITICAL: Fit scaler only on real (non-padded) timesteps
    # This prevents padded zeros from skewing statistics
    scaler.fit(X_flat[~mask_flat])
    # Transform all data (including padded rows for consistency)
    X_flat = scaler.transform(X_flat)
    X = X_flat.reshape(X.shape)
    
    # Save artifacts for inference
    # Use the global feature columns collected from all sequences
    feature_cols_final = feature_cols_global
    
    np.save(EXPORT_DIR / "feature_cols.npy", feature_cols_final)
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(EXPORT_DIR / "gesture_classes.npy", gesture_classes)
    np.save(EXPORT_DIR / "use_feature_engineering.npy", CONFIG['use_feature_engineering'])
    np.save(EXPORT_DIR / "pad_mask.npy", pad_mask)  # Save padding mask for inference
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
    
    # Determine dimensions for models
    # Count different feature types
    n_features = X.shape[-1]
    
    # Determine dimensions from feature columns (more reliable than heuristics)
    imu_cols = [c for c in feature_cols_final if any(c.startswith(p) for p in 
                ['linear_acc', 'angular_vel', 'acc_', 'rot_', 'acc_mag', 'acc_jerk', 'angular_distance'])]
    thm_cols = [c for c in feature_cols_final if c.startswith('thm_')]
    tof_cols = [c for c in feature_cols_final if c.startswith('tof_')]
    
    # Use actual counts if available, otherwise fall back to estimates
    if imu_cols or thm_cols or tof_cols:
        imu_dim = len(imu_cols) if imu_cols else 6
        thm_dim = max(1, len(thm_cols))  # Ensure at least 1 to avoid Conv1d(0, ...)
        tof_dim = max(1, len(tof_cols))
        # Adjust if total doesn't match
        total_assigned = imu_dim + thm_dim + tof_dim
        if total_assigned != n_features:
            # Distribute remainder to TOF
            tof_dim = max(1, n_features - imu_dim - thm_dim)
    else:
        # Fallback heuristic if no clear prefixes
        imu_dim = min(13, n_features // 2) if CONFIG['use_feature_engineering'] else 7
        remaining = n_features - imu_dim
        thm_dim = max(1, min(4, remaining // 2))  # Ensure at least 1
        tof_dim = max(1, remaining - thm_dim)
    
    print(f"Feature dimensions: Total={n_features}, IMU={imu_dim}, THM={thm_dim}, TOF={tof_dim}")
    
    # Cross-validation training
    sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['seed'])
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, subjects)):
        print(f"\n{'='*40}")
        print(f"FOLD {fold + 1}/{CONFIG['n_splits']}")
        print(f"{'='*40}")
        
        # Initialize fresh models for this fold
        models = {
            'gated': GatedTwoBranchModel(
                imu_dim=imu_dim,
                tof_thm_dim=tof_dim + thm_dim,  # Combine TOF and THM for this model
                n_classes=n_classes
            ),
            'mixed': TwoBranchMixedModel(
                imu_dim=imu_dim,
                tof_dim=tof_dim + thm_dim,  # Combine TOF and THM
                n_classes=n_classes
            ),
            'bert': CMIBertModel(
                imu_dim=imu_dim,
                tof_dim=tof_dim,
                thm_dim=thm_dim,
                n_classes=n_classes,
                bert_layers=12,  # Increased from 8 for better performance
                bert_heads=8
            )
        }
        
        # Initialize weights properly for better training
        for name, model in models.items():
            # Initialize weights
            for m in model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.ones_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.LSTM, nn.GRU)):
                    for param in m.parameters():
                        if len(param.shape) >= 2:
                            nn.init.orthogonal_(param)
                        else:
                            nn.init.zeros_(param)
        
        # Print model parameters (only on first fold)
        if fold == 0:
            for name, model in models.items():
                params = sum(p.numel() for p in model.parameters())
                print(f"{name.upper()} Model: {params:,} parameters")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create datasets and loaders with padding masks
        train_pad_mask = pad_mask[train_idx] if 'pad_mask' in locals() else None
        val_pad_mask = pad_mask[val_idx] if 'pad_mask' in locals() else None
        
        train_dataset = CMIDataset(X_train, y_train, pad_mask=train_pad_mask)
        val_dataset = CMIDataset(X_val, y_val, pad_mask=val_pad_mask)
        
        # Use drop_last=True for training to avoid BatchNorm issues with single-sample batches
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        
        # Train each model
        for model_name, model in models.items():
            print(f"\nTraining {model_name.upper()} model...")
            
            model = model.to(device)
            
            # Calculate class weights for balanced training
            class_weights = compute_class_weight('balanced', 
                                                 classes=np.unique(y_train), 
                                                 y=y_train)
            class_weights = torch.FloatTensor(class_weights).to(device)
            
            # Use label smoothing only for gated model as per original notebook
            label_smoothing = CONFIG['label_smoothing'] if model_name == 'gated' else 0.0
            criterion = nn.CrossEntropyLoss(weight=class_weights, 
                                           label_smoothing=label_smoothing)
            # Use standard Adam like original notebook (not AdamW)
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr_init'], 
                                  weight_decay=CONFIG['weight_decay'])
            # Use single cosine annealing like original notebook
            scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
            scaler = GradScaler(enabled=torch.cuda.is_available())
            
            # Initialize early stopping variables for this model
            best_val_score = -1.0  # Track best F1 score, not loss
            patience_counter = 0
            
            for epoch in range(CONFIG['epochs']):
                # Train
                is_gated = (model_name == 'gated')
                train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                        scaler, device, is_gated, CONFIG['masking_prob'], CONFIG['mixup_alpha'],
                                        imu_dim=imu_dim if is_gated else None, thm_dim=thm_dim if is_gated else 0)
                
                # Validate
                val_loss, val_score = validate_epoch(model, val_loader, criterion, 
                                                     device, is_gated, label_encoder)
                
                scheduler.step()
                
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{CONFIG['epochs']} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Score: {val_score:.4f}, "
                      f"LR: {current_lr:.6f}")
                
                # Early stopping based on F1 score (not loss)
                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                    # Save best model by F1 score
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'pad_len': pad_len,
                        'imu_dim': imu_dim,
                        'tof_dim': tof_dim,
                        'thm_dim': thm_dim,
                        'n_classes': n_classes,
                        'feature_engineering': CONFIG['use_feature_engineering'],
                        'best_val_score': best_val_score
                    }, EXPORT_DIR / f"{model_name}_fold{fold}.pth")
                    print(f"  → New best F1: {best_val_score:.4f} (saved)")
                else:
                    patience_counter += 1
                    if patience_counter >= CONFIG['patience']:
                        print(f"Early stopping at epoch {epoch+1}")
                        print(f"Best validation F1 score: {best_val_score:.4f}")
                        break
        
        # Report fold performance summary
        print(f"\nFold {fold + 1} Summary:")
        for model_name in models.keys():
            checkpoint_path = EXPORT_DIR / f"{model_name}_fold{fold}.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'best_val_score' in checkpoint:
                    print(f"  {model_name.upper()}: Best F1 = {checkpoint['best_val_score']:.4f}")
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("Training completed! Models saved to:", EXPORT_DIR)
    print("="*60)


# ==================== Main Execution ====================

if __name__ == "__main__":
    if TRAIN:
        train_models()
    else:
        # Inference mode - load pretrained models
        print("▶ INFERENCE MODE - Loading pretrained 3-model ensemble for submission")
        print(f"  Loading artifacts from: {PRETRAINED_DIR}")
        
        # Load saved artifacts
        final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
        pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
        scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)
        use_feature_engineering = bool(np.load(PRETRAINED_DIR / "use_feature_engineering.npy"))
        
        print(f"  Loaded configuration:")
        print(f"    - Sequence length: {pad_len}")
        print(f"    - Feature engineering: {use_feature_engineering}")
        print(f"    - Number of features: {len(final_feature_cols)}")
        print(f"    - Gesture classes: {gesture_classes}")
        
        # Determine dimensions from feature columns (match training logic exactly)
        imu_cols = [c for c in final_feature_cols if any(c.startswith(p) for p in 
                    ['linear_acc', 'angular_vel', 'acc_', 'rot_', 'acc_mag', 'acc_jerk', 'angular_distance'])]
        thm_cols = [c for c in final_feature_cols if c.startswith('thm_')]
        tof_cols = [c for c in final_feature_cols if c.startswith('tof_')]
        
        imu_dim = len(imu_cols) if imu_cols else 7  # Default fallback
        thm_dim = len(thm_cols) if thm_cols else 1
        tof_dim = len(tof_cols) if tof_cols else 5
        
        print(f"  Feature dimensions: IMU={imu_dim}, THM={thm_dim}, TOF={tof_dim}")
        
        # Sanity check: Print first 5 feature columns to verify artifacts
        print(f"  First 5 features: {final_feature_cols[:5] if len(final_feature_cols) >= 5 else final_feature_cols}")
        
        # Load models from all folds for ensemble
        all_models = {'gated': [], 'mixed': [], 'bert': []}
        
        # Try to load models from all folds
        for fold in range(CONFIG['n_splits']):
            for model_name in ['gated', 'mixed', 'bert']:
                checkpoint_path = PRETRAINED_DIR / f"{model_name}_fold{fold}.pth"
                
                if checkpoint_path.exists():
                    print(f"  Loading {model_name} model from fold {fold}...")
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # Extract dimensions from checkpoint
                    ckpt_imu_dim = checkpoint.get('imu_dim', imu_dim)
                    ckpt_tof_dim = checkpoint.get('tof_dim', tof_dim)
                    ckpt_thm_dim = checkpoint.get('thm_dim', thm_dim)
                    ckpt_n_classes = checkpoint.get('n_classes', len(gesture_classes))
                    ckpt_pad_len = checkpoint.get('pad_len', pad_len)
                    
                    if model_name == 'gated':
                        # Use new signature for consistency
                        model = GatedTwoBranchModel(
                            imu_dim=ckpt_imu_dim,
                            tof_thm_dim=ckpt_tof_dim + ckpt_thm_dim,  # Combine TOF and THM
                            n_classes=ckpt_n_classes
                        )
                    elif model_name == 'mixed':
                        model = TwoBranchMixedModel(
                            imu_dim=ckpt_imu_dim,
                            tof_dim=ckpt_tof_dim + ckpt_thm_dim,
                            n_classes=ckpt_n_classes
                        )
                    else:  # bert
                        model = CMIBertModel(
                            imu_dim=ckpt_imu_dim,
                            tof_dim=ckpt_tof_dim,
                            thm_dim=ckpt_thm_dim,
                            n_classes=ckpt_n_classes,
                            bert_layers=12,  # Match training (increased from 8)
                            bert_heads=8
                        )
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    model.eval()
                    all_models[model_name].append(model)
        
        # Check if we have models (CRITICAL for submission)
        total_models = sum(len(models) for models in all_models.values())
        print(f"\n  CRITICAL CHECK - Loaded {total_models} models total:")
        for model_name, models in all_models.items():
            if models:
                print(f"    - {model_name}: {len(models)} fold(s)")
            else:
                print(f"    - {model_name}: WARNING - NO MODELS LOADED!")
        
        if total_models == 0:
            raise RuntimeError("No models found! Please check PRETRAINED_DIR path.")
        
        # Create ensemble prediction function
        def predict(sequence, demographics=None):
            """Make prediction for a single sequence - Kaggle expects 'predict' function"""
            # demographics parameter is kept for API compatibility but not used
            # Convert if polars DataFrame
            if pl and hasattr(sequence, 'to_pandas'):
                df = sequence.to_pandas()
            else:
                df = sequence
            
            # Apply feature engineering if needed
            if use_feature_engineering:
                df = engineer_features(df.copy())
            
            # Prepare data
            df_filled = df[final_feature_cols].ffill().bfill().fillna(0)
            X = df_filled.values.astype(np.float32)
            orig_len = len(X)  # Store original length before padding
            
            # CRITICAL FIX: Pad FIRST, then scale (matching training)
            # This ensures padded zeros are also normalized like during training
            X = pad_sequences_torch([X], pad_len)[0]  # (pad_len, features)
            X = scaler.transform(X).astype(np.float32)  # Scale after padding
            X = torch.from_numpy(X).unsqueeze(0).to(device)  # (1, pad_len, features)
            
            # Create padding mask for BERT (True where padded)
            pad_mask_np = np.zeros((pad_len,), dtype=bool)
            if orig_len < pad_len:
                pad_mask_np[orig_len:] = True
            pad_mask_t = torch.from_numpy(pad_mask_np).unsqueeze(0).to(device)  # (1, pad_len)
            
            # Collect LOGITS (not softmax) from each model family
            logits_by_family = {'gated': [], 'mixed': [], 'bert': []}
            
            # Get predictions from all models
            for model_name, models in all_models.items():
                if not models:
                    continue
                    
                for model in models:
                    model.eval()
                    with torch.no_grad():
                        # Apply TTA for gated and mixed models
                        if model_name in ('gated', 'mixed') and CONFIG['tta_steps'] > 1:
                            logit_accum = 0
                            for _ in range(CONFIG['tta_steps']):
                                noise = torch.randn_like(X) * CONFIG['tta_noise_stddev']
                                X_aug = X + noise
                                if model_name == 'gated':
                                    out, _ = model(X_aug)
                                else:  # mixed
                                    out = model(X_aug)
                                logit_accum += out
                            output = logit_accum / CONFIG['tta_steps']
                        else:
                            # No TTA for BERT or if TTA disabled
                            if model_name == 'gated':
                                output, _ = model(X)
                            elif model_name == 'bert':
                                output = model(X, pad_mask=pad_mask_t)  # Pass mask to BERT
                            else:
                                output = model(X)
                        
                        # Store raw logits (not softmax)
                        logits_by_family[model_name].append(output.squeeze(0).cpu().numpy())
            
            # IMPROVED: Ensemble in logit space (softmax after weighted sum)
            ensemble_weights = CONFIG['ensemble_weights']  # [gated, mixed, bert]
            model_order = ['gated', 'mixed', 'bert']
            
            accumulated_logits = 0
            for i, model_name in enumerate(model_order):
                if logits_by_family[model_name]:
                    # Average logits within each model family
                    family_mean_logits = np.mean(logits_by_family[model_name], axis=0)
                    # Apply ensemble weight to logits
                    accumulated_logits += ensemble_weights[i] * family_mean_logits
            
            if isinstance(accumulated_logits, np.ndarray):
                # Apply softmax once at the end
                final_probs = np.exp(accumulated_logits - np.max(accumulated_logits))
                final_probs /= final_probs.sum()
                idx = int(final_probs.argmax())
                return str(gesture_classes[idx])
            else:
                # Fallback to most common class
                return str(gesture_classes[0])
        
        print("\n  Models loaded successfully - ready for inference!")
        
        # For Kaggle submission
        if 'kaggle_evaluation' in sys.modules or os.path.exists('/kaggle'):
            import kaggle_evaluation.cmi_inference_server
            inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
            
            if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                inference_server.serve()
            else:
                # Local testing
                inference_server.run_local_gateway(
                    data_paths=(
                        RAW_DIR / 'test.csv',
                        RAW_DIR / 'test_demographics.csv',
                    )
                )