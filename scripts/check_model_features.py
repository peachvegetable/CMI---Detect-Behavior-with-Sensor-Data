#!/usr/bin/env python3
"""
Check what features the saved model expects
"""
import torch
import os

# Get the root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Try to load a checkpoint
checkpoint_path = os.path.join(PROJECT_ROOT, 'improved_model_fold_0.pth')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("=" * 60)
    print("MODEL CHECKPOINT ANALYSIS")
    print("=" * 60)
    
    # Check config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("\nTraining Config:")
        print(f"  use_angular_velocity: {config.get('use_angular_velocity', False)}")
        print(f"  use_angular_distance: {config.get('use_angular_distance', False)}")
        print(f"  use_statistical_features: {config.get('use_statistical_features', False)}")
    
    # Check feature columns
    if 'feature_cols' in checkpoint:
        feature_cols = checkpoint['feature_cols']
        print(f"\nTotal features saved: {len(feature_cols)}")
        
        # Categorize features
        linear_acc_cols = [c for c in feature_cols if 'linear_acc' in c]
        acc_cols = [c for c in feature_cols if c.startswith('acc_') and 'linear' not in c]
        rot_cols = [c for c in feature_cols if c.startswith('rot_')]
        thm_cols = [c for c in feature_cols if c.startswith('thm_')]
        tof_cols = [c for c in feature_cols if c.startswith('tof_')]
        other_cols = [c for c in feature_cols if c not in linear_acc_cols + acc_cols + rot_cols + thm_cols + tof_cols]
        
        print(f"\nFeature breakdown:")
        print(f"  Linear acceleration: {len(linear_acc_cols)} - {linear_acc_cols[:5]}...")
        print(f"  Raw acceleration: {len(acc_cols)} - {acc_cols[:5]}...")
        print(f"  Rotation: {len(rot_cols)} - {rot_cols[:5]}...")
        print(f"  Thermopile: {len(thm_cols)}")
        print(f"  Time-of-Flight: {len(tof_cols)}")
        print(f"  Other engineered: {len(other_cols)} - {other_cols}")
    
    # Check model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Check first layer shapes to understand feature expectations
        imu_layer = 'imu_encoder.0.weight'
        tof_layer = 'tof_encoder.0.weight'
        
        if imu_layer in state_dict:
            imu_shape = state_dict[imu_layer].shape
            print(f"\nIMU encoder expects: {imu_shape[1]} channels")
            
        if tof_layer in state_dict:
            tof_shape = state_dict[tof_layer].shape
            print(f"TOF encoder expects: {tof_shape[1]} channels")
            
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("The model was trained BEFORE adding gravity removal features.")
    print("We need to either:")
    print("1. Retrain the model with new features, OR")
    print("2. Revert the submission script to match the trained model")
    
else:
    print(f"No checkpoint found at {checkpoint_path}")
    print("Please train the model first!")