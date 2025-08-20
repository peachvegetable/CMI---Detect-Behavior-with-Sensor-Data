#!/usr/bin/env python3
"""
Diagnostic script to identify issues with submission predictions
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# BFRB gestures
BFRB_GESTURES = [
    'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
    'Eyebrow - pull hair', 'Eyelash - pull hair', 'Neck - pinch skin',
    'Neck - scratch', 'Cheek - pinch skin'
]

def analyze_predictions():
    """Analyze model predictions on validation data"""
    
    print("=" * 80)
    print("SUBMISSION DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    # Check for saved models
    print("\n1. Checking for saved models...")
    model_files = []
    for fold in range(5):
        # Check in project root first
        model_path = os.path.join(PROJECT_ROOT, f'improved_model_fold_{fold}.pth')
        if not os.path.exists(model_path):
            # Fallback to models directory
            model_path = os.path.join(MODEL_DIR, f'improved_model_fold_{fold}.pth')
        
        if os.path.exists(model_path):
            model_files.append(model_path)
            print(f"   ✓ Found: {os.path.basename(model_path)}")
        else:
            print(f"   ✗ Missing: improved_model_fold_{fold}.pth")
    
    if not model_files:
        print("\n   ERROR: No model files found! Train models first.")
        return
    
    # Load a model to check its contents
    print(f"\n2. Analyzing model checkpoint...")
    checkpoint = torch.load(model_files[0], map_location='cpu', weights_only=False)
    
    print(f"   - Model keys: {list(checkpoint.keys())[:5]}...")
    
    if 'label_encoder' in checkpoint:
        label_encoder = checkpoint['label_encoder']
        print(f"   - Number of classes: {len(label_encoder.classes_)}")
        print(f"   - BFRB classes found: {sum(1 for c in label_encoder.classes_ if c in BFRB_GESTURES)}/8")
        
        # Check if all BFRB gestures are in the label encoder
        missing_bfrb = [g for g in BFRB_GESTURES if g not in label_encoder.classes_]
        if missing_bfrb:
            print(f"   - WARNING: Missing BFRB gestures: {missing_bfrb}")
    
    if 'score' in checkpoint:
        print(f"   - Training score: {checkpoint['score']:.4f}")
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"   - Binary threshold in training: 0.5 (hardcoded)")
        print(f"   - IMU-only train prob: {config.get('imu_only_train_prob', 'N/A')}")
        print(f"   - IMU-only val prob: {config.get('imu_only_val_prob', 'N/A')}")
    
    # Check feature columns
    if 'feature_cols' in checkpoint:
        feature_cols = checkpoint['feature_cols']
        print(f"   - Number of features: {len(feature_cols)}")
        
        # Check feature ordering
        imu_features = [f for f in feature_cols if f.startswith(('linear_acc_', 'rot_', 'ang_vel', 'angular'))]
        thm_features = [f for f in feature_cols if f.startswith('thm_')]
        tof_features = [f for f in feature_cols if f.startswith('tof_')]
        
        print(f"   - IMU features: {len(imu_features)}")
        print(f"   - Thermopile features: {len(thm_features)}")
        print(f"   - ToF features: {len(tof_features)}")
    
    # Analyze prediction distribution
    print("\n3. Analyzing prediction behavior...")
    
    # Check if we have validation predictions saved
    val_preds_path = os.path.join(MODEL_DIR, 'validation_predictions.npz')
    if os.path.exists(val_preds_path):
        print("   Loading validation predictions...")
        val_data = np.load(val_preds_path)
        
        if 'binary_probs' in val_data:
            binary_probs = val_data['binary_probs']
            print(f"   - Binary probabilities shape: {binary_probs.shape}")
            print(f"   - Binary prob distribution:")
            print(f"     Mean: {np.mean(binary_probs[:, 1]):.3f}")
            print(f"     Std: {np.std(binary_probs[:, 1]):.3f}")
            print(f"     Min: {np.min(binary_probs[:, 1]):.3f}")
            print(f"     Max: {np.max(binary_probs[:, 1]):.3f}")
            
            # Check different thresholds
            print("\n   - Predictions at different thresholds:")
            for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                n_positive = np.sum(binary_probs[:, 1] > thresh)
                pct = 100 * n_positive / len(binary_probs)
                print(f"     Threshold {thresh}: {n_positive}/{len(binary_probs)} ({pct:.1f}%) positive")
    
    # Test feature engineering consistency
    print("\n4. Testing feature engineering...")
    
    # Load a small sample of training data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), nrows=1000)
    
    # Get unique sequences
    sequences = train_df.groupby('sequence_id').size()
    print(f"   - Sample sequences: {len(sequences)}")
    print(f"   - Sequence lengths: min={sequences.min()}, max={sequences.max()}, mean={sequences.mean():.1f}")
    
    # Check for NaN values in key columns
    print("\n5. Data quality checks...")
    key_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    for col in key_cols:
        if col in train_df.columns:
            nan_count = train_df[col].isna().sum()
            if nan_count > 0:
                print(f"   - WARNING: {col} has {nan_count} NaN values")
    
    # Check gesture distribution
    if 'gesture' in train_df.columns:
        gesture_counts = train_df['gesture'].value_counts()
        bfrb_count = sum(gesture_counts.get(g, 0) for g in BFRB_GESTURES)
        total_count = len(train_df)
        print(f"\n   - BFRB samples: {bfrb_count}/{total_count} ({100*bfrb_count/total_count:.1f}%)")
    
    print("\n6. Key findings:")
    print("   - The 0.5 threshold for binary classification might be too high")
    print("   - Consider adjusting threshold based on validation performance")
    print("   - Check if 'Text on phone' is the correct default for non-BFRB")
    print("   - Ensure model paths are correctly set for Kaggle submission")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_predictions()