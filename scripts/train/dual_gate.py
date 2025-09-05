#!/usr/bin/env python3

import os
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence

from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from tqdm import tqdm
import gc

# Import polars for inference if available
try:
    import polars as pl
except ImportError:
    pl = None

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
    'batch_size': 32,
    'pad_percentile': 95,
    'lr_init': 5e-4,
    'weight_decay': 3e-3,
    'mixup_alpha': 0.4,
    'epochs': 160, 
    'patience': 40,
    'n_splits': 5,
    'masking_prob': 0.25,
    'gate_loss_weight': 0.2,
    'label_smoothing': 0.1,
    'tta_steps': 10,  # Test-time augmentation steps
    'tta_noise_stddev': 0.01,  # Noise std for TTA
    'device': device,
    'seed': 42
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
    RAW_DIR = Path("data")
    EXPORT_DIR = Path("./dual_gate_model")
    EXPORT_DIR.mkdir(exist_ok=True)
    from competition_metric import CompetitionMetric
    print(f"\nTRAIN MODE - Starting training")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Device: {CONFIG['device']}")
else:
    # For Kaggle submission
    RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
    PRETRAINED_DIR = Path("/kaggle/input/dual_gate/pytorch/default/4")
    print("▶ INFERENCE MODE - Loading pretrained models for submission")

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
        except ValueError:
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
        except ValueError: 
            pass
    return angular_vel

def calculate_angular_distance(rot_data):
    """Calculate frame-to-frame angular distance"""
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_dist = np.zeros(len(quat_values))
    
    for i in range(len(quat_values) - 1):
        q1, q2 = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q1)) or np.all(np.isnan(q2)): 
            continue
        try:
            r1, r2 = R.from_quat(q1), R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angular_dist[i] = np.linalg.norm(relative_rotation.as_rotvec())
        except ValueError: 
            pass
    return angular_dist

# ==================== PyTorch Model Components ====================

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
    """Residual block with SE attention"""
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

class AttentionLayer(nn.Module):
    """Attention mechanism for sequence aggregation"""
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

class GatedTwoBranchModel(nn.Module):
    def __init__(self, imu_dim, tof_dim, n_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim
        
        # IMU branch (deep)
        self.imu_branch = nn.Sequential(
            ResidualSEBlock(imu_dim, 64, kernel_size=3, dropout=0.1),
            ResidualSEBlock(64, 128, kernel_size=5, dropout=0.1)
        )
        
        # TOF/THM branch (lighter)
        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_dim, 64, kernel_size=3, padding=1, bias=False),
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
            nn.Linear(tof_dim, 16),
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
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
        
        # Split input into IMU and TOF/THM
        imu_data = x[:, :self.imu_dim, :]
        tof_data = x[:, self.imu_dim:, :]
        
        # Process IMU branch
        imu_features = self.imu_branch(imu_data)
        
        # Process TOF/THM branch with gating
        tof_features_base = self.tof_branch(tof_data)
        # Feed gate from raw TOF/THM data (matches original TF implementation)
        gate_logits = self.gate_network(tof_data)  # (batch, 1)
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
            noise = torch.randn_like(combined) * 0.09
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

# ==================== Dataset and Data Loading ====================

class BFRBDataset(Dataset):
    """PyTorch dataset for BFRB detection with gate labels"""
    def __init__(self, sequences, labels, gate_labels=None, transform=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.gate_labels = torch.FloatTensor(gate_labels) if gate_labels is not None else torch.ones(len(labels))
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx]
        y = self.labels[idx]
        gate = self.gate_labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y, gate


def mixup_data(x, y, gate, alpha=1.0):
    """Mixup augmentation with gate labels"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    gate_a, gate_b = gate, gate[index]
    
    return mixed_x, y_a, y_b, gate_a, gate_b, lam

# ==================== Loss Functions ====================

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, n_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        loss = torch.sum(-true_dist * pred, dim=-1)
        
        if self.weight is not None:
            loss = loss * self.weight[target]
        
        return torch.mean(loss)

# ==================== Training Functions ====================

def train_epoch(model, train_loader, main_criterion, gate_criterion, optimizer, device, 
                use_mixup=True, mixup_alpha=0.4, masking_prob=0.25, imu_dim=13, 
                gate_loss_weight=0.2):
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_gate_loss = 0
    total_correct = 0
    total_samples = 0
    
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target, gate_target_orig) in enumerate(progress_bar):
        data = data.to(device)
        target = target.to(device)
        
        # Use sensor-based gate labels (has TOF/THM=1, no TOF/THM=0)
        gate_target = torch.ones(data.size(0), device=device)
        if masking_prob > 0:
            mask = torch.rand(data.size(0)) < masking_prob
            mask = mask.to(device)
            data[mask, imu_dim:, :] = 0
            gate_target[mask] = 0.0
        
        # Apply mixup (note: mixup doesn't change lengths)
        if use_mixup and np.random.random() < 0.5:
            data, target_a, target_b, gate_a, gate_b, lam = mixup_data(
                data, target, gate_target, mixup_alpha
            )
            
            output, gate_logits = model(data)
            main_loss = lam * main_criterion(output, target_a) + (1 - lam) * main_criterion(output, target_b)
            gate_loss = lam * gate_criterion(gate_logits, gate_a) + (1 - lam) * gate_criterion(gate_logits, gate_b)
        else:
            output, gate_logits = model(data)
            main_loss = main_criterion(output, target)
            gate_loss = gate_criterion(gate_logits, gate_target)
        
        # Combined loss
        loss = main_loss + gate_loss_weight * gate_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_gate_loss += gate_loss.item()
        
        # Calculate accuracy (for monitoring)
        _, predicted = output.max(1)
        if not use_mixup or np.random.random() >= 0.5:
            total_correct += predicted.eq(target).sum().item()
            total_samples += target.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'main': f'{main_loss.item():.4f}',
            'gate': f'{gate_loss.item():.4f}'
        })
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return total_loss / len(train_loader), accuracy

def validate(model, val_loader, main_criterion, gate_criterion, device, label_encoder=None, 
             imu_dim=13, simulate_test=True, gate_loss_weight=0.2):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_gates = []
    
    
    with torch.no_grad():
        for data, target, _ in tqdm(val_loader, desc='Validating'):
            data, target = data.to(device), target.to(device)
            
            # Track which samples have TOF/THM masked
            gate_target = torch.ones(data.size(0), device=device)
            
            # Original notebook does NOT mask during validation
            # Only mask if explicitly requested (for test simulation)
            if simulate_test and False:  # Disabled to match original
                mask = torch.rand(data.size(0)) < 0.5
                mask = mask.to(device)
                data[mask, imu_dim:, :] = 0
                gate_target[mask] = 0.0
            
            output, gate_logits = model(data)
            main_loss = main_criterion(output, target)
            gate_loss = gate_criterion(gate_logits, gate_target)
            loss = main_loss + gate_loss_weight * gate_loss
            
            total_loss += loss.item()
            
            # Simple prediction without gated decision
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            
            all_targets.extend(target.cpu().numpy())
            all_gates.extend(torch.sigmoid(gate_logits).cpu().numpy())
    
    # Calculate metrics
    if label_encoder is not None:
        y_true = label_encoder.inverse_transform(all_targets)
        y_pred = label_encoder.inverse_transform(all_preds)
        
        metric = CompetitionMetric()
        score = metric.calculate_hierarchical_f1(
            pd.DataFrame({'gesture': y_true}),
            pd.DataFrame({'gesture': y_pred})
        )
    else:
        score = f1_score(all_targets, all_preds, average='macro')
    
    # Print gate statistics
    gate_array = np.array(all_gates)
    print(f"  Gate stats: mean={gate_array.mean():.3f}, >0.5: {(gate_array > 0.5).mean():.1%}")
    
    # Debug info removed - no gated decision
    
    return total_loss / len(val_loader), score

# ==================== Main Training Script ====================

def train_models():
    print("\nLoading and preparing data...")
    df = pd.read_csv(RAW_DIR / "train.csv")
    
    # Label encoding
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)
    
    # Physics-based feature engineering
    print("  Applying physics-based feature engineering...")
    linear_accel_list = []
    angular_vel_list = []
    angular_dist_list = []
    
    for _, group in tqdm(df.groupby('sequence_id'), desc="Processing sequences"):
        # Remove gravity
        linear_accel = remove_gravity_from_acc(
            group[['acc_x', 'acc_y', 'acc_z']], 
            group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        )
        linear_accel_df = pd.DataFrame(
            linear_accel, 
            columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], 
            index=group.index
        )
        linear_accel_list.append(linear_accel_df)
        
        # Angular velocity
        angular_vel = calculate_angular_velocity_from_quat(
            group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        )
        angular_vel_df = pd.DataFrame(
            angular_vel, 
            columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], 
            index=group.index
        )
        angular_vel_list.append(angular_vel_df)
        
        # Angular distance
        angular_dist = calculate_angular_distance(
            group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        )
        angular_dist_df = pd.DataFrame(
            angular_dist, 
            columns=['angular_distance'], 
            index=group.index
        )
        angular_dist_list.append(angular_dist_df)
    
    # Concatenate new features
    df = pd.concat([df, pd.concat(linear_accel_list)], axis=1)
    df = pd.concat([df, pd.concat(angular_vel_list)], axis=1)
    df = pd.concat([df, pd.concat(angular_dist_list)], axis=1)
    
    # Additional features
    df['linear_acc_mag'] = np.sqrt(
        df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2
    )
    df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    
    # Define feature columns
    imu_cols_base = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z'] + \
                    [c for c in df.columns if c.startswith('rot_')]
    imu_engineered = ['linear_acc_mag', 'linear_acc_mag_jerk', 
                      'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 
                      'angular_distance']
    imu_cols = list(dict.fromkeys(imu_cols_base + imu_engineered))
    
    thm_cols = [c for c in df.columns if c.startswith('thm_')]
    
    # Aggregate TOF features
    tof_aggregated_cols = []
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        if all(col in df.columns for col in pixel_cols):
            tof_data = df[pixel_cols].replace(-1, np.nan)
            df[f'tof_{i}_mean'] = tof_data.mean(axis=1)
            df[f'tof_{i}_std'] = tof_data.std(axis=1)
            df[f'tof_{i}_min'] = tof_data.min(axis=1)
            df[f'tof_{i}_max'] = tof_data.max(axis=1)
            tof_aggregated_cols.extend([f'tof_{i}_mean', f'tof_{i}_std', 
                                       f'tof_{i}_min', f'tof_{i}_max'])
    
    final_feature_cols = imu_cols + thm_cols + tof_aggregated_cols
    imu_dim = len(imu_cols)
    tof_thm_dim = len(thm_cols) + len(tof_aggregated_cols)
    
    print(f"  IMU features: {imu_dim}")
    print(f"  THM + TOF features: {tof_thm_dim}")
    print(f"  Total features: {len(final_feature_cols)}")
    
    # Save feature columns for inference
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))
    
    # Build sequences
    print("\nBuilding sequences...")
    sequences = []
    labels = []
    groups = []
    lengths = []
    
    for seq_id, seq_df in tqdm(df.groupby('sequence_id'), desc="Creating sequences"):
        seq_features = seq_df[final_feature_cols].ffill().bfill().fillna(0).values
        sequences.append(seq_features)
        labels.append(seq_df['gesture_int'].iloc[0])
        groups.append(seq_df['subject'].iloc[0] if 'subject' in seq_df.columns else seq_id)
        lengths.append(len(seq_features))
    
    # Fit scaler
    print("  Fitting StandardScaler...")
    all_data = np.vstack(sequences)
    scaler = StandardScaler().fit(all_data)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
    
    # Scale and pad sequences
    sequences_scaled = [scaler.transform(seq) for seq in sequences]
    pad_len = int(np.percentile(lengths, CONFIG['pad_percentile']))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    
    # Pad sequences
    sequences_padded = []
    for seq in sequences_scaled:
        if len(seq) > pad_len:
            seq = seq[:pad_len]
        elif len(seq) < pad_len:
            pad_width = ((0, pad_len - len(seq)), (0, 0))
            seq = np.pad(seq, pad_width, mode='constant', constant_values=0)
        sequences_padded.append(seq)
    
    X = np.array(sequences_padded)
    # Transpose for Conv1d (batch, time, features) -> (batch, features, time)
    X = X.transpose(0, 2, 1)
    y = np.array(labels)
    groups = np.array(groups)
    
    print(f"  Final shape: {X.shape}")
    print(f"  Sequence length: {pad_len}")
    
    # Cross-validation
    print("\n" + "="*60)
    print("STARTING 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['seed'])
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"\n===== FOLD {fold+1}/{CONFIG['n_splits']} =====")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        # Create datasets and loaders
        train_dataset = BFRBDataset(X_train, y_train)
        val_dataset = BFRBDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Initialize model
        model = GatedTwoBranchModel(
            imu_dim=imu_dim,
            tof_dim=tof_thm_dim,
            n_classes=len(le.classes_)
        ).to(CONFIG['device'])
        
        # Compute class weights for balanced training
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(len(le.classes_)),
            y=y_train
        )
        class_weights = torch.FloatTensor(class_weights).to(CONFIG['device'])
        
        # Loss functions with class weights
        main_criterion = LabelSmoothingCrossEntropy(
            n_classes=len(le.classes_),
            smoothing=CONFIG['label_smoothing'],
            weight=class_weights
        )
        gate_criterion = nn.BCEWithLogitsLoss()  # For numerical stability
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['lr_init'],
            weight_decay=CONFIG['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.5,
            patience=3
        )
        
        # Training loop
        best_score = 0
        patience_counter = 0
        
        for epoch in range(CONFIG['epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, main_criterion, gate_criterion, 
                optimizer, CONFIG['device'],
                use_mixup=True, 
                mixup_alpha=CONFIG['mixup_alpha'],
                masking_prob=CONFIG['masking_prob'],
                imu_dim=imu_dim,
                gate_loss_weight=CONFIG['gate_loss_weight']
            )
            
            # Validate
            val_loss, val_score = validate(
                model, val_loader, main_criterion, gate_criterion, CONFIG['device'],
                label_encoder=le,
                imu_dim=imu_dim,
                simulate_test=True,
                gate_loss_weight=CONFIG['gate_loss_weight']
            )
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_score)
            
            # Save best model
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'fold': fold,
                    'epoch': epoch,
                    'score': best_score,
                    'config': CONFIG
                }, EXPORT_DIR / f"model_fold_{fold}.pth")
                print(f"✓ Saved best model (score: {best_score:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        cv_scores.append(best_score)
        print(f"\nFold {fold+1} best score: {best_score:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Fold Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"Mean CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


# ==================== Prediction Function for Inference ====================

def predict(sequence, demographics) -> str:
    """Prediction function for Kaggle submission with TTA and 95th percentile padding"""
    # Convert if polars DataFrame
    if pl and hasattr(sequence, 'to_pandas'):
        df_seq = sequence.to_pandas()
    else:
        df_seq = sequence
    
    # Physics-based feature engineering
    linear_accel = remove_gravity_from_acc(df_seq, df_seq)
    df_seq['linear_acc_x'], df_seq['linear_acc_y'], df_seq['linear_acc_z'] = linear_accel[:, 0], linear_accel[:, 1], linear_accel[:, 2]
    df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2)
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
    
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq['angular_vel_x'], df_seq['angular_vel_y'], df_seq['angular_vel_z'] = angular_vel[:, 0], angular_vel[:, 1], angular_vel[:, 2]
    df_seq['angular_distance'] = calculate_angular_distance(df_seq)
    
    # Aggregate TOF features
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        if all(col in df_seq.columns for col in pixel_cols):
            tof_data = df_seq[pixel_cols].replace(-1, np.nan)
            df_seq[f'tof_{i}_mean'] = tof_data.mean(axis=1)
            df_seq[f'tof_{i}_std'] = tof_data.std(axis=1)
            df_seq[f'tof_{i}_min'] = tof_data.min(axis=1)
            df_seq[f'tof_{i}_max'] = tof_data.max(axis=1)
    
    # Prepare features
    mat_unscaled = df_seq[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
    mat_scaled = scaler.transform(mat_unscaled)
    
    # Pad sequence
    if len(mat_scaled) > pad_len:
        mat_scaled = mat_scaled[:pad_len]
    elif len(mat_scaled) < pad_len:
        pad_width = ((0, pad_len - len(mat_scaled)), (0, 0))
        mat_scaled = np.pad(mat_scaled, pad_width, mode='constant', constant_values=0)
    
    # Transpose for Conv1d
    pad_input = torch.FloatTensor(mat_scaled).unsqueeze(0).transpose(1, 2)  # (1, features, time)
    
    # TTA loop
    all_tta_predictions = []
    for tta_step in range(CONFIG['tta_steps']):
        # Add noise for TTA (except first step)
        if CONFIG['tta_steps'] > 1 and tta_step > 0:
            noisy_input = pad_input + torch.randn_like(pad_input) * CONFIG['tta_noise_stddev']
        else:
            noisy_input = pad_input
        
        noisy_input = noisy_input.to(CONFIG['device'])
        
        # Ensemble prediction across folds
        all_fold_predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                # Get main output (ignore gate output)
                main_preds, _ = model(noisy_input)
                all_fold_predictions.append(main_preds.cpu().numpy())
        
        # Average across folds
        avg_fold_prediction = np.mean(all_fold_predictions, axis=0)
        all_tta_predictions.append(avg_fold_prediction)
    
    # Average across TTA steps
    final_avg_prediction = np.mean(all_tta_predictions, axis=0)
    
    # Get predicted class
    idx = int(final_avg_prediction.argmax())
    return str(gesture_classes[idx])

# ==================== Main Execution ====================

if __name__ == "__main__":
    if TRAIN:
        train_models()
    else:
        # Inference mode - load pretrained models
        print("INFERENCE MODE loading artifacts from", PRETRAINED_DIR)
        
        # Load saved artifacts
        final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
        pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
        scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)
        
        # Determine dimensions from feature columns
        imu_cols = [c for c in final_feature_cols if c.startswith(('linear_acc', 'rot_', 'angular'))]
        thm_cols = [c for c in final_feature_cols if c.startswith('thm_')]
        tof_cols = [c for c in final_feature_cols if c.startswith('tof_')]
        imu_dim = len(imu_cols)
        tof_thm_dim = len(thm_cols) + len(tof_cols)
        
        # Load models
        models = []
        print(f"  Loading {CONFIG['n_splits']} models for ensemble inference...")
        for fold in range(CONFIG['n_splits']):
            checkpoint = torch.load(
                PRETRAINED_DIR / f"model_fold_{fold}.pth",
                map_location=CONFIG['device']
            )
            model = GatedTwoBranchModel(
                imu_dim=imu_dim,
                tof_dim=tof_thm_dim,
                n_classes=len(gesture_classes)
            ).to(CONFIG['device'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        
        print("  Models, scaler, feature_cols, pad_len loaded – ready for evaluation")
        
        # For Kaggle submission
        import kaggle_evaluation.cmi_inference_server
        inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
        
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            inference_server.serve()
        else:
            inference_server.run_local_gateway(
                data_paths=(
                    RAW_DIR / 'test.csv',
                    RAW_DIR / 'test_demographics.csv',
                )
            )