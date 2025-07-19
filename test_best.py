#!/usr/bin/env python3
"""
BFRB Detection Testing Script
=============================
Test trained models on new data
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import HybridModel  # Use HybridModel alias for compatibility
from train_best import CONFIG


def load_models(model_prefix='hybrid_model_fold_', n_folds=5):
    """Load all trained models"""
    models = []
    scalers = []
    label_encoders = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold in range(n_folds):
        model_path = f'{model_prefix}{fold}.pth'
        if not os.path.exists(model_path):
            print(f"⚠️  Model {model_path} not found")
            continue
            
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model
        model = HybridModel(n_classes=18).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append(model)
        scalers.append(checkpoint['scaler'])
        
        # Handle missing label_encoder in old checkpoints
        if 'label_encoder' in checkpoint:
            label_encoders.append(checkpoint['label_encoder'])
        else:
            # Create a default label encoder with gesture classes
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            gesture_classes = [
                'Above ear - pull hair', 'Cheek - pinch skin', 'Drink from bottle/cup',
                'Eyebrow - pull hair', 'Eyelash - pull hair', 
                'Feel around in tray and pull out an object', 'Forehead - pull hairline',
                'Forehead - scratch', 'Glasses on/off', 'Neck - pinch skin',
                'Neck - scratch', 'Pinch knee/leg skin', 'Pull air toward your face',
                'Scratch knee/leg skin', 'Text on phone', 'Wave hello',
                'Write name in air', 'Write name on leg'
            ]
            le.classes_ = np.array(gesture_classes)
            label_encoders.append(le)
        
        print(f"✅ Loaded fold {fold} (score: {checkpoint.get('score', 'N/A'):.4f})")
    
    return models, scalers, label_encoders, device


def predict_sequence(models, scalers, label_encoders, sequence_data, feature_cols):
    """Make prediction for a single sequence"""
    predictions_binary = []
    predictions_multi = []
    
    device = next(models[0].parameters()).device
    
    for model, scaler in zip(models, scalers):
        # Extract features
        features = sequence_data[feature_cols].values.astype(np.float32)
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)
        
        # Create windows
        if len(features) >= CONFIG['window_size']:
            # Use last window
            window = features[-CONFIG['window_size']:]
        else:
            # Pad if too short
            padding = CONFIG['window_size'] - len(features)
            window = np.pad(features, ((0, padding), (0, 0)), mode='constant')
        
        # Normalize
        window_scaled = scaler.transform(window)
        
        # Convert to tensor
        window_tensor = torch.FloatTensor(window_scaled.T).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(window_tensor)
            binary_probs = F.softmax(outputs['binary'], dim=1)
            multi_probs = F.softmax(outputs['multiclass'], dim=1)
            
            predictions_binary.append(binary_probs.cpu().numpy())
            predictions_multi.append(multi_probs.cpu().numpy())
    
    # Average predictions
    avg_binary = np.mean(predictions_binary, axis=0)
    avg_multi = np.mean(predictions_multi, axis=0)
    
    # Get final predictions
    is_bfrb = avg_binary[0, 1] > 0.5
    gesture_idx = np.argmax(avg_multi[0])
    gesture = label_encoders[0].inverse_transform([gesture_idx])[0]
    
    return {
        'gesture': gesture,
        'is_bfrb': is_bfrb,
        'binary_confidence': avg_binary[0, 1],
        'multi_confidence': np.max(avg_multi[0]),
        'binary_probs': avg_binary[0],
        'multi_probs': avg_multi[0]
    }


def test_on_validation_data():
    """Test on a portion of training data to verify model performance"""
    print("\n📊 Testing on validation subset...")
    
    # Load models
    models, scalers, label_encoders, device = load_models()
    if not models:
        print("❌ No models found!")
        return
    
    # Load data
    df = pd.read_csv('data/train.csv')
    gesture_df = df[df['behavior'] == 'Performs gesture'].copy()
    if len(gesture_df) == 0:
        gesture_df = df[df['gesture'].notna()].copy()
    
    # Feature columns
    feature_cols = [col for col in gesture_df.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
    
    # Sample some sequences for testing
    test_sequences = gesture_df['sequence_id'].unique()[-100:]  # Last 100 sequences
    
    predictions = []
    true_labels = []
    
    print(f"Testing on {len(test_sequences)} sequences...")
    
    for seq_id in test_sequences:
        seq_data = gesture_df[gesture_df['sequence_id'] == seq_id]
        true_gesture = seq_data['gesture'].iloc[0]
        
        # Predict
        pred = predict_sequence(models, scalers, label_encoders, seq_data, feature_cols)
        
        predictions.append(pred['gesture'])
        true_labels.append(true_gesture)
    
    # Calculate accuracy
    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
    print(f"\nValidation Accuracy: {accuracy:.2%}")
    
    # F1 scores
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)
    pred_encoded = le.transform(predictions)
    
    f1_macro = f1_score(true_encoded, pred_encoded, average='macro')
    print(f"Macro F1 Score: {f1_macro:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))


def test_on_test_csv():
    """Test on the actual test.csv file"""
    print("\n🔍 Testing on test.csv...")
    
    # Load models
    models, scalers, label_encoders, device = load_models()
    if not models:
        print("❌ No models found!")
        return
    
    # Load test data
    test_df = pd.read_csv('data/test.csv')
    print(f"Test shape: {test_df.shape}")
    print(f"Test sequences: {test_df['sequence_id'].nunique()}")
    
    # Feature columns
    feature_cols = [col for col in test_df.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
    
    # Process each sequence
    results = []
    
    for seq_id in test_df['sequence_id'].unique():
        seq_data = test_df[test_df['sequence_id'] == seq_id]
        
        # Predict
        pred = predict_sequence(models, scalers, label_encoders, seq_data, feature_cols)
        
        results.append({
            'sequence_id': seq_id,
            'gesture': pred['gesture'],
            'is_bfrb': pred['is_bfrb'],
            'binary_confidence': pred['binary_confidence'],
            'multi_confidence': pred['multi_confidence']
        })
        
        print(f"Sequence {seq_id}: {pred['gesture']} "
              f"(BFRB: {pred['is_bfrb']}, "
              f"Binary conf: {pred['binary_confidence']:.3f}, "
              f"Multi conf: {pred['multi_confidence']:.3f})")
    
    # Save predictions
    results_df = pd.DataFrame(results)
    results_df[['sequence_id', 'gesture']].to_csv('test_predictions.csv', index=False)
    
    print(f"\n✅ Predictions saved to test_predictions.csv")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"Total predictions: {len(results_df)}")
    print(f"BFRB predictions: {results_df['is_bfrb'].sum()} / {len(results_df)}")
    print(f"Mean binary confidence: {results_df['binary_confidence'].mean():.3f}")
    print(f"Mean multi confidence: {results_df['multi_confidence'].mean():.3f}")
    
    print(f"\nPredicted gestures:")
    print(results_df['gesture'].value_counts())


def analyze_predictions():
    """Analyze prediction confidence and distribution"""
    print("\n📈 Analyzing predictions...")
    
    if not os.path.exists('test_predictions.csv'):
        print("❌ No predictions found. Run test_on_test_csv() first.")
        return
    
    # Load models to get probabilities
    models, scalers, label_encoders, device = load_models()
    if not models:
        return
    
    # Load test data
    test_df = pd.read_csv('data/test.csv')
    feature_cols = [col for col in test_df.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
    
    # Collect detailed predictions
    all_binary_probs = []
    all_multi_probs = []
    
    for seq_id in test_df['sequence_id'].unique():
        seq_data = test_df[test_df['sequence_id'] == seq_id]
        pred = predict_sequence(models, scalers, label_encoders, seq_data, feature_cols)
        
        all_binary_probs.append(pred['binary_probs'])
        all_multi_probs.append(pred['multi_probs'])
    
    # Plot confidence distributions
    plt.figure(figsize=(12, 5))
    
    # Binary confidence
    plt.subplot(1, 2, 1)
    binary_confidences = [p[1] for p in all_binary_probs]
    plt.hist(binary_confidences, bins=20, alpha=0.7, color='blue')
    plt.xlabel('BFRB Probability')
    plt.ylabel('Count')
    plt.title('Binary Classification Confidence Distribution')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision threshold')
    plt.legend()
    
    # Multiclass confidence
    plt.subplot(1, 2, 2)
    multi_confidences = [np.max(p) for p in all_multi_probs]
    plt.hist(multi_confidences, bins=20, alpha=0.7, color='green')
    plt.xlabel('Max Class Probability')
    plt.ylabel('Count')
    plt.title('Multiclass Classification Confidence Distribution')
    
    plt.tight_layout()
    plt.savefig('prediction_confidence_distribution.png', dpi=150)
    plt.close()
    
    print("✅ Confidence distribution plot saved to prediction_confidence_distribution.png")
    
    # Print statistics
    print(f"\nBinary confidence stats:")
    print(f"  Mean: {np.mean(binary_confidences):.3f}")
    print(f"  Std: {np.std(binary_confidences):.3f}")
    print(f"  Min: {np.min(binary_confidences):.3f}")
    print(f"  Max: {np.max(binary_confidences):.3f}")
    
    print(f"\nMulticlass confidence stats:")
    print(f"  Mean: {np.mean(multi_confidences):.3f}")
    print(f"  Std: {np.std(multi_confidences):.3f}")
    print(f"  Min: {np.min(multi_confidences):.3f}")
    print(f"  Max: {np.max(multi_confidences):.3f}")


def main():
    """Main testing function"""
    print("🔍 BFRB DETECTION TESTING")
    print("=" * 60)
    
    # Test on validation data to verify model performance
    test_on_validation_data()
    
    # Test on actual test.csv
    test_on_test_csv()
    
    # Analyze predictions
    analyze_predictions()
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()