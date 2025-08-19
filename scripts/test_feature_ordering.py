#!/usr/bin/env python3
"""
Test script to verify feature ordering fix
"""

import sys
import os

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

# Configuration matching the training script
CONFIG = {
    'use_angular_velocity': True,
    'use_angular_distance': True,
    'use_statistical_features': True,
}

def test_feature_ordering():
    """Test that feature ordering is correct"""
    print("Testing feature ordering fix...")
    
    # Create a dummy dataframe with all expected columns
    columns = []
    
    # Add linear acceleration columns
    columns.extend(['linear_acc_x', 'linear_acc_y', 'linear_acc_z'])
    
    # Add rotation columns
    columns.extend(['rot_w', 'rot_x', 'rot_y', 'rot_z'])
    
    # Add thermopile columns
    columns.extend([f'thm_{i}' for i in range(5)])
    
    # Add ToF columns (example)
    columns.extend([f'tof_{i}' for i in range(50)])
    
    # Add engineered feature columns that will be created
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
    
    columns.extend(engineered_cols)
    
    # Add some other columns that should NOT be included
    columns.extend(['sequence_id', 'timestamp', 'gesture'])
    
    # Create dummy dataframe
    df = pd.DataFrame(np.random.randn(100, len(columns)), columns=columns)
    
    # Apply the new feature ordering logic
    # Collect IMU base features
    imu_base_cols = []
    imu_base_cols += [col for col in df.columns if col.startswith('linear_acc_')]
    imu_base_cols += [col for col in df.columns if col.startswith('rot_')]
    
    # Collect thermopile and ToF features
    thm_cols = [col for col in df.columns if col.startswith('thm_')]
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    
    # Build feature_cols in the correct order for the model
    feature_cols = imu_base_cols + engineered_cols + thm_cols + tof_cols
    
    # Print results
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"IMU features: {len(imu_base_cols)} base + {len(engineered_cols)} engineered = {len(imu_base_cols) + len(engineered_cols)}")
    print(f"Thermopile features: {len(thm_cols)}")
    print(f"ToF features: {len(tof_cols)}")
    
    # Verify ordering
    print("\nFeature ordering (first 30):")
    for i, col in enumerate(feature_cols[:30]):
        print(f"  {i:3d}: {col}")
    
    # Calculate expected indices for model slicing
    n_imu_total = len(imu_base_cols) + len(engineered_cols)
    n_thm = len(thm_cols)
    n_tof = len(tof_cols)
    
    print(f"\nModel slicing indices:")
    print(f"  IMU data: x[:, :{n_imu_total}, :]")
    print(f"  THM data: x[:, {n_imu_total}:{n_imu_total + n_thm}, :]")
    print(f"  TOF data: x[:, {n_imu_total + n_thm}:, :]")
    
    # Verify the slicing would work correctly
    print(f"\nVerifying slices capture correct features:")
    imu_slice = feature_cols[:n_imu_total]
    thm_slice = feature_cols[n_imu_total:n_imu_total + n_thm]
    tof_slice = feature_cols[n_imu_total + n_thm:]
    
    print(f"  IMU slice contains only IMU features: {all(col.startswith(('linear_acc_', 'rot_')) or col in engineered_cols for col in imu_slice)}")
    print(f"  THM slice contains only THM features: {all(col.startswith('thm_') for col in thm_slice)}")
    print(f"  TOF slice contains only TOF features: {all(col.startswith('tof_') for col in tof_slice)}")
    
    print("\n✓ Feature ordering test complete!")
    return feature_cols

if __name__ == "__main__":
    test_feature_ordering()