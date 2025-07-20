# CMI - Detect Behavior with Sensor Data Competition Summary

## Competition Overview
- **Goal**: Distinguish body-focused repetitive behaviors (BFRBs) from non-BFRB gestures using multimodal sensor data
- **Device**: Helios wrist-worn device with IMU, thermopile, and time-of-flight sensors
- **Evaluation**: Average of binary F1 (BFRB vs non-BFRB) and macro F1 (specific gestures)

## Dataset Analysis
- **Total samples**: 574,945
- **Gesture performance samples**: 255,817
- **Unique sequences**: 8,150
- **Subjects**: 81
- **Target behaviors**: 8 BFRB gestures, 10 non-BFRB gestures (18 total)
- **Sensor features**: 332 (7 IMU + 5 thermopile + 320 TOF)

## Current Best Results
Based on the hybrid model training:
- **Cross-validation scores**: [0.835, 0.867, 0.852, 0.851, 0.870]
- **Mean score**: 0.855 ± 0.012
- **Best score**: 0.870

## Model Architecture (HybridModel)
1. **Sensor-specific encoders**:
   - IMU: Conv1D layers with residual blocks
   - Thermopile: Conv1D with squeeze-excitation
   - TOF: Spatial processing with dimension reduction

2. **Feature fusion**:
   - Multi-scale residual blocks
   - Global context attention
   - Adaptive pooling

3. **Dual-pathway classification**:
   - Binary pathway for BFRB detection
   - Multiclass pathway for specific gestures

## Training Configuration
- **Window size**: 120 timesteps
- **Stride**: 20 (overlapping windows)
- **Batch size**: 48
- **Learning rate**: 8e-4 with CosineAnnealing
- **Regularization**: Label smoothing (0.15), Focal loss (γ=2.5), Mixup (α=0.3)

## Key Insights
1. **Data preprocessing**:
   - Focus on "Performs gesture" phase for cleaner signals
   - Aggressive data augmentation with overlapping windows
   - Robust scaling for sensor normalization

2. **Model design**:
   - Sensor-specific processing is crucial
   - Attention mechanisms improve performance
   - Multi-task learning (binary + multiclass) is effective

3. **Training strategies**:
   - Cross-validation essential for robust evaluation
   - Early stopping with patience prevents overfitting
   - Class weighting handles imbalance

## Test Performance
- Local test on 2 sample sequences shows model is working
- Binary confidence: ~43%
- Class confidence: ~24%
- Predictions align with expected BFRB/non-BFRB classification

## Recommendations for Improvement
1. **Ensemble methods**: Combine predictions from multiple folds
2. **Post-processing**: Threshold optimization for binary classification
3. **Feature engineering**: Extract statistical features from sensor data
4. **Advanced architectures**: Transformer-based models, graph neural networks
5. **Data augmentation**: Time warping, sensor noise injection

## Competition Submission
- Use the trained models with Kaggle evaluation API
- Ensure handling of IMU-only sequences in test set
- Optimize inference speed for real-time predictions