"""
Unified preprocessing module to ensure consistency between training and inference
"""

import numpy as np
import pandas as pd
import polars as pl


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


class UnifiedFeatureEngineer:
    """Unified feature engineering that works with both pandas and polars"""
    
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
    def engineer_features_from_arrays(acc_data, rot_data, window_size=10):
        """
        Apply feature engineering to numpy arrays
        Returns a dictionary of engineered features
        """
        features = {}
        
        # Angular velocity
        angular_vel = UnifiedFeatureEngineer.calculate_angular_velocity(rot_data)
        features['ang_vel_x'] = angular_vel[:, 0]
        features['ang_vel_y'] = angular_vel[:, 1]
        features['ang_vel_z'] = angular_vel[:, 2]
        
        # Angular distance
        features['angular_distance'] = UnifiedFeatureEngineer.calculate_angular_distance(rot_data)
        
        # Jerk
        jerk = UnifiedFeatureEngineer.calculate_jerk(acc_data)
        features['jerk_x'] = jerk[:, 0]
        features['jerk_y'] = jerk[:, 1]
        features['jerk_z'] = jerk[:, 2]
        
        # Acceleration magnitude
        features['acc_magnitude'] = np.linalg.norm(acc_data, axis=1)
        
        # MAD features
        acc_mad = UnifiedFeatureEngineer.calculate_mad(acc_data, window_size)
        features['acc_mad_x'] = acc_mad[:, 0]
        features['acc_mad_y'] = acc_mad[:, 1]
        features['acc_mad_z'] = acc_mad[:, 2]
        
        # Rotation angle from quaternion
        features['rotation_angle'] = 2 * np.arccos(np.clip(rot_data[:, 0], -1, 1))
        
        return features
    
    @staticmethod
    def engineer_features_pandas(df, window_size=10):
        """Apply feature engineering to pandas dataframe"""
        engineered_features = []
        
        for seq_id in df['sequence_id'].unique():
            seq_data = df[df['sequence_id'] == seq_id].copy()
            
            # Extract base features
            acc_data = seq_data[['acc_x', 'acc_y', 'acc_z']].values
            rot_data = seq_data[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
            
            # Get engineered features
            features = UnifiedFeatureEngineer.engineer_features_from_arrays(
                acc_data, rot_data, window_size
            )
            
            # Add to dataframe
            for feature_name, feature_values in features.items():
                seq_data[feature_name] = feature_values
            
            engineered_features.append(seq_data)
            
        return pd.concat(engineered_features, ignore_index=True)
    
    @staticmethod
    def engineer_features_polars(df, window_size=10):
        """Apply feature engineering to polars dataframe"""
        # Convert to pandas for processing
        df_pd = df.to_pandas()
        
        # Extract base features
        acc_data = df_pd[['acc_x', 'acc_y', 'acc_z']].values
        rot_data = df_pd[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
        
        # Get engineered features
        features = UnifiedFeatureEngineer.engineer_features_from_arrays(
            acc_data, rot_data, window_size
        )
        
        # Add to dataframe
        for feature_name, feature_values in features.items():
            df_pd[feature_name] = feature_values
        
        # Convert back to polars
        return pl.from_pandas(df_pd)


def get_feature_columns(config):
    """Get list of feature columns based on configuration"""
    # Base feature columns
    base_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z',
                 'thm_0', 'thm_1', 'thm_2', 'thm_3', 'thm_4']
    
    # Add ToF columns
    tof_cols = [f'tof_{i}' for i in range(64)]
    base_cols.extend(tof_cols)
    
    # Add engineered columns based on config
    engineered_cols = []
    if config.get('use_angular_velocity', True):
        engineered_cols.extend(['ang_vel_x', 'ang_vel_y', 'ang_vel_z'])
    if config.get('use_angular_distance', True):
        engineered_cols.append('angular_distance')
    if config.get('use_statistical_features', True):
        engineered_cols.extend(['jerk_x', 'jerk_y', 'jerk_z', 'acc_magnitude',
                               'acc_mad_x', 'acc_mad_y', 'acc_mad_z', 'rotation_angle'])
    
    return base_cols + engineered_cols