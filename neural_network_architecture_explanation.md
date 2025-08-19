# Detailed Neural Network Architecture Explanation for BFRB Detection

## Overview
This model is designed to detect Body-Focused Repetitive Behaviors (BFRBs) using multi-sensor data. Here's a comprehensive breakdown:

## Overall Architecture Philosophy
The model uses a **multi-stream CNN architecture** with attention mechanisms to process three different sensor types separately before fusing them. This is crucial because:
- Different sensors capture fundamentally different information (motion vs temperature vs distance)
- Each sensor type has different characteristics requiring specialized processing
- The model learns sensor-specific features before combining them

## Core Components Explained

### 1. Conv1D Layers (1D Convolutional Neural Networks)
- **What it does**: Applies filters that slide across time-series data to detect patterns
- **Why 1D instead of 2D**: Your sensor data is sequential (time-based), not images
- **Example**: `nn.Conv1d(64, 128, kernel_size=7, padding=3)`
  - Takes 64 input channels, outputs 128 channels
  - kernel_size=7 means it looks at 7 time steps at once to detect patterns
  - padding=3 maintains sequence length

### 2. Squeeze-and-Excitation (SE) Blocks (lines 206-222)
- **Purpose**: Channel attention mechanism - learns which features are important
- **How it works**:
  1. Global average pooling compresses each channel to a single value
  2. Two fully connected layers learn channel importance weights
  3. Sigmoid ensures weights are between 0-1
  4. Multiplies original features by these weights
- **Why use it**: Helps the model focus on relevant sensor channels dynamically

### 3. Residual Blocks (lines 225-250)
- **Purpose**: Enables training of deeper networks by solving the vanishing gradient problem
- **Structure**:
  - Two Conv1D layers with BatchNorm and ReLU
  - SE attention block
  - Skip connection that adds input directly to output
- **Why important**: Allows information to flow directly through shortcuts, making deep networks trainable

### 4. Three Sensor-Specific Encoders

#### IMU Encoder (lines 276-282)
- Processes motion data (accelerometer + gyroscope)
- Larger kernel (7) to capture longer motion patterns
- Deeper architecture (2 residual blocks) for complex motion features

#### Thermopile Encoder (lines 285-290)
- Processes temperature sensor data
- Smaller network (32→64 channels) as temperature patterns are simpler
- kernel_size=5 for medium-range temporal patterns

#### Time-of-Flight Encoder (lines 293-302)
- Processes distance measurements
- Starts with 1x1 conv (pointwise) to expand features
- Then 5x5 conv for temporal patterns
- Most complex encoder due to high-dimensional ToF data

### 5. Feature Fusion Layers (lines 306-314)
- Concatenates all sensor features (320 channels total)
- Progressive expansion: 320→512→768→1024 channels
- **AdaptiveAvgPool1d**: Dynamically reduces sequence length while preserving information
- Why progressive expansion: Allows learning of increasingly complex multi-sensor patterns

### 6. Attention Mechanism (lines 317-321, 406-441)
- **Purpose**: Identifies which time steps are most important
- **Process**:
  1. Conv layers compress 1024→128→1 to get importance score per time step
  2. Tanh activation for bounded attention values
  3. Softmax normalizes scores to sum to 1
  4. Masks padding tokens in variable-length sequences
- **Why crucial**: Not all time steps are equally important for behavior detection

### 7. Dual-Task Classification Heads

#### Binary Head (lines 332-342)
- Determines if behavior is BFRB or not
- Progressive reduction: 768→384→192→2
- Higher dropout (0.5→0.4→0.3) for regularization

#### Multiclass Head (lines 345-355)
- Identifies specific BFRB type (18 classes)
- Larger architecture: 768→512→256→18
- Separate pathway allows specialized learning for each task

## Key Design Decisions & Why They Matter

1. **Multi-Stream Architecture**: Each sensor type gets specialized processing before fusion
   - Respects the different nature of each sensor's data

2. **Dynamic Sequence Handling**: Uses attention masking for variable-length sequences
   - Real-world sensor data has varying durations

3. **Hierarchical Feature Learning**: 
   - Early layers: Low-level patterns (sudden movements, temperature changes)
   - Middle layers: Sensor fusion and temporal relationships
   - Late layers: High-level behavior patterns

4. **Regularization Techniques**:
   - Dropout (prevents overfitting)
   - BatchNorm (stabilizes training)
   - Label smoothing (improves generalization)

5. **Dual-Task Learning**: Binary + multiclass classification
   - Binary task provides strong gradient signal
   - Helps model learn general BFRB patterns before specializing

## Why This Architecture for BFRB Detection

1. **Temporal Patterns**: Conv1D excels at detecting temporal patterns in sensor data
2. **Multi-Scale Features**: Different kernel sizes capture patterns at different time scales
3. **Attention for Key Moments**: BFRBs have characteristic moments the attention mechanism can focus on
4. **Sensor Fusion**: Combines motion, temperature, and distance for comprehensive behavior understanding
5. **Deep but Trainable**: Residual connections allow deep network without gradient issues

This architecture is specifically tailored for time-series sensor data classification, making it ideal for detecting subtle behavioral patterns from wearable sensors.

## Technical Details for Interview

### Activation Functions
- **ReLU**: Used after convolutional layers for non-linearity and to avoid vanishing gradients
- **Tanh**: Used in attention mechanism to bound attention values between -1 and 1
- **Sigmoid**: Used in SE blocks to create gates (0-1 values) for channel importance
- **Softmax**: Used in final attention weights to ensure they sum to 1

### Normalization Techniques
- **BatchNorm1d**: Normalizes activations across the batch dimension
  - Stabilizes training by reducing internal covariate shift
  - Allows higher learning rates
  - Acts as a regularizer

### Loss Functions
- **Focal Loss**: Addresses class imbalance in binary classification
  - Down-weights easy examples, focuses on hard examples
  - Gamma parameter controls the focus strength
- **Label Smoothing**: Prevents overconfidence in predictions
  - Replaces hard targets (0,1) with soft targets (0.1, 0.9)
  - Improves generalization

### Training Techniques
- **Mixed Precision Training**: Uses both float16 and float32 for efficiency
- **Gradient Clipping**: Prevents exploding gradients in deep networks
- **Learning Rate Scheduling**: CosineAnnealingWarmRestarts for adaptive learning
- **Data Augmentation**: Mixup augmentation blends samples for better generalization

### Model Capacity
- Total parameters: ~10-15 million (estimated)
- Designed to run on edge devices with GPU support
- Inference time: <100ms per sample on modern GPUs

This architecture represents state-of-the-art design principles for time-series classification, combining ideas from computer vision (residual networks, attention) with domain-specific adaptations for sensor data processing.