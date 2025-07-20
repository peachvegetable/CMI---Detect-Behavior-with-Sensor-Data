# CMI Project Report: Advanced BFRB Detection System

## Summary

This report outlines the development of an advanced Body-Focused Repetitive Behavior (BFRB) detection system as part of the CMI project. The project implements a sophisticated machine learning pipeline capable of identifying and classifying repetitive behaviors using multi-modal sensor data from IMU (Inertial Measurement Unit), thermopile, and time-of-flight sensors.

## Project Overview

### Objective
Develop a robust machine learning model to detect and classify Body-Focused Repetitive Behaviors (BFRBs) such as hair pulling, skin picking, and scratching using wearable sensor data.

### Target Behaviors
The system is designed to detect 8 specific BFRB gestures:
- Above ear - pull hair
- Forehead - pull hairline  
- Forehead - scratch
- Eyebrow - pull hair
- Eyelash - pull hair
- Neck - pinch skin
- Neck - scratch
- Cheek - pinch skin

## Detailed Feature Engineering Analysis

### 1. Angular Velocity Calculation from Quaternions

**Data Format:**
The sensor data contains complete quaternions with four components: `rot_w`, `rot_x`, `rot_y`, `rot_z`, representing unit quaternions [w, x, y, z] that describe the device's 3D orientation at each timestep.

**Mathematical Foundation:**
The angular velocity extraction from quaternion sequences represents a critical innovation for capturing rotational motion patterns characteristic of BFRB behaviors.

**Detailed Implementation Process:**
```
Given consecutive quaternions:
q₁ = [w₁, x₁, y₁, z₁] at time t
q₂ = [w₂, x₂, y₂, z₂] at time t+Δt

Step 1: Calculate quaternion difference
q_diff = q₂ * q₁* (where q₁* is the conjugate of q₁)
q₁* = [w₁, -x₁, -y₁, -z₁]

Step 2: Extract rotation angle
θ = 2 * arccos(|q_diff.w|)

Step 3: Compute rotation axis (if θ > 0)
axis = [q_diff.x, q_diff.y, q_diff.z] / sin(θ/2)

Step 4: Calculate angular velocity
ω = axis * θ / Δt
```

**Quaternion Mathematics Implementation:**
```python
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return [w, x, y, z]

def quaternion_conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]
```

**Why Quaternion-Based Angular Velocity Matters:**
- **Behavioral Signature Detection:** BFRB gestures involve specific rotational patterns (e.g., wrist rotation during hair pulling, head tilting during eyebrow pulling)
- **Temporal Dynamics:** Captures the instantaneous rate of rotational change, distinguishing between slow positioning movements and rapid repetitive motions
- **Noise Robustness:** Quaternion-based calculations are more stable than Euler angles, avoiding gimbal lock issues and providing smooth angular velocity profiles
- **High-Frequency Motion Detection:** 50Hz sampling allows detection of subtle rotational patterns that distinguish different BFRB types

### 2. Cumulative Angular Distance Tracking

**Mathematical Implementation:**
```
For consecutive quaternions q_i and q_{i+1}:
1. Quaternion dot product: dot = q_i · q_{i+1} = w_i*w_{i+1} + x_i*x_{i+1} + y_i*y_{i+1} + z_i*z_{i+1}
2. Angular distance: θ_i = 2 * arccos(|dot|)
3. Cumulative distance: D_i = D_{i-1} + θ_i
```

**Physical Interpretation:**
The dot product between unit quaternions gives the cosine of half the angle between orientations. Taking 2 * arccos(|dot|) gives the actual angular distance traveled between consecutive orientations.

**Implementation Details:**
```python
def calculate_angular_distance(rotation_data):
    angular_dist = np.zeros(len(rotation_data))
    
    for i in range(1, len(rotation_data)):
        q1 = rotation_data[i-1]  # [w, x, y, z]
        q2 = rotation_data[i]    # [w, x, y, z]
        
        # Quaternion dot product
        dot = np.clip(np.dot(q1, q2), -1, 1)  # Clipping prevents numerical errors
        
        # Angular distance between orientations
        angle = 2 * np.arccos(np.abs(dot))
        angular_dist[i] = angular_dist[i-1] + angle
        
    return angular_dist
```

**Behavioral Relevance:**
- **Repetitive Motion Detection:** Accumulates total rotational movement, identifying sustained repetitive behaviors
- **Gesture Boundary Detection:** Sharp increases in cumulative distance indicate gesture initiation/termination
- **Intensity Quantification:** Higher cumulative distances correlate with more intense BFRB episodes

### 3. Jerk Analysis (Third Derivative of Position)

**Calculation Method:**
```
Given acceleration vectors a(t):
Jerk = da/dt ≈ (a_{t+1} - a_t) / Δt
```

**Neuromotor Significance:**
- **Motor Control Patterns:** Jerk profiles reveal underlying motor control strategies
- **Smoothness Metrics:** BFRB gestures often exhibit characteristic jerk patterns due to their compulsive nature
- **Tremor Detection:** Distinguishes intentional movements from tremor-like behaviors

**Clinical Relevance:** Jerk analysis captures the "abruptness" of movements, with BFRB gestures typically showing higher jerk magnitudes than normal activities.

### 4. Statistical Feature Engineering with Sliding Windows

**Mean Absolute Deviation (MAD) Implementation:**
```
For window W of size n:
MAD = (1/n) * Σ|x_i - μ_W|
where μ_W is the window mean
```

**Multi-Scale Analysis:**
- **Short-term patterns (window = 10):** Captures immediate movement characteristics
- **Medium-term trends:** Reveals sustained behavioral patterns
- **Temporal consistency:** Identifies repetitive micro-patterns within gestures

### 5. Acceleration Magnitude and Rotational Angle Features

**Magnitude Calculation:**
```
|a| = √(ax² + ay² + az²)
Rotation angle: θ = 2 * arccos(|qw|)
```

**Behavioral Interpretation:**
- **Gesture Intensity:** Magnitude captures the "force" behind movements
- **Orientation Tracking:** Rotation angles reveal hand/wrist positioning preferences
- **Energy Expenditure:** Higher magnitudes correlate with more vigorous BFRB episodes

## Model Architecture Analysis

### 1. Multi-Modal Sensor Processing Strategy

**Architectural Philosophy:**
The model employs separate specialized encoders for each sensor modality, recognizing that different sensors capture complementary aspects of behavior:

**IMU Encoder (Enhanced with Engineered Features):**
- **Input Dimensionality:** 7 base + up to 12 engineered features = 19 total channels
- **Convolutional Architecture:** Conv1d(19→64) → ResidualBlock(64→128) → ResidualBlock(128→128)
- **Rationale:** IMU data requires temporal convolution to capture motion dynamics
- **Kernel Size:** 7-point kernels capture short-term temporal dependencies critical for gesture recognition

**Thermopile Encoder:**
- **Input:** 5 temperature sensor channels
- **Architecture:** Conv1d(5→32) → ResidualBlock(32→64)
- **Purpose:** Temperature patterns indicate skin contact and friction, key indicators of skin-focused BFRBs
- **Smaller Network:** Fewer parameters due to simpler thermal patterns

**Time-of-Flight (ToF) Encoder:**
- **Input:** Variable number of distance sensors
- **Architecture:** Conv1d(n→128) → Conv1d(128→64) → ResidualBlock(64→128) → ResidualBlock(128→128)
- **Function:** Captures spatial proximity patterns, detecting when hands approach face/head regions

### 2. Squeeze-and-Excitation (SE) Attention Mechanism

**Mathematical Formulation:**
```
1. Global Average Pooling: z = (1/T) * Σ x_t
2. Channel Reduction: s₁ = ReLU(W₁ * z)
3. Channel Expansion: s₂ = Sigmoid(W₂ * s₁)
4. Feature Recalibration: x' = x ⊙ s₂
```

**Why SE Blocks Improve BFRB Detection:**
- **Channel Importance Learning:** Automatically learns which sensor channels are most relevant for each behavior type
- **Adaptive Feature Selection:** Different BFRB gestures may rely on different sensor combinations
- **Noise Suppression:** Downweights irrelevant channels, reducing false positives

### 3. Residual Architecture with Skip Connections

**Residual Block Design:**
```
F(x) = ReLU(BN(Conv(ReLU(BN(Conv(x))))) + x
```

**Benefits for Temporal Sequence Modeling:**
- **Gradient Flow:** Enables training of deeper networks without vanishing gradients
- **Feature Preservation:** Skip connections preserve both low-level sensor readings and high-level behavioral patterns
- **Temporal Coherence:** Maintains consistency across time steps in variable-length sequences

**BFRB-Specific Advantages:**
- **Multi-Resolution Features:** Combines fine-grained sensor details with broader behavioral patterns
- **Robust Training:** Enables stable training on limited BFRB datasets
- **Transfer Learning:** Pre-trained features can transfer across different behavior types

### 4. Global Context Attention Mechanism

**Implementation:**
```
1. Context Extraction: C = Tanh(Conv1d(features))
2. Attention Weights: α = Softmax(Conv1d(C))
3. Weighted Features: F' = Σ(α_t * F_t)
```

**Behavioral Significance:**
- **Temporal Dependencies:** BFRB gestures often have preparatory phases followed by execution
- **Context Modeling:** Understands that gesture meaning depends on surrounding temporal context
- **Variable Length Handling:** Adaptively weights important time steps regardless of sequence length

### 5. Multi-Task Learning Architecture

**Dual-Head Design:**
- **Binary Head:** BFRB vs. non-BFRB classification
- **Multiclass Head:** Specific gesture type identification
- **Shared Backbone:** Common feature extraction layers

**Loss Function Strategy:**
```
L_total = 0.6 * L_binary + 0.4 * L_multiclass
```

**Why This Weighting Works:**
- **Competition Metric Alignment:** Matches the evaluation criteria
- **Hierarchical Learning:** Binary classification provides strong supervision for multiclass learning
- **Feature Sharing:** Common representations benefit both tasks

**Accuracy Benefits:**
- **Improved Generalization:** Multi-task learning acts as regularization
- **Better Feature Learning:** Shared representations capture more robust behavioral patterns
- **Enhanced Robustness:** Binary task provides stability when multiclass becomes challenging

### 6. Dynamic Sequence Processing Innovation

**Technical Challenge:**
Real-world sensor sequences have variable lengths (ranging from seconds to minutes), but traditional models require fixed-length inputs.

**Solution Implementation:**
- **Per-Batch Padding:** Dynamic padding to batch maximum length instead of global maximum
- **Adaptive Pooling:** AdaptiveAvgPool1d adjusts to actual sequence lengths
- **Custom Collate Function:** Efficient batching without excessive padding

### 7. Advanced Loss Function Design

**Focal Loss for Binary Classification:**
```
FL = -α(1-p_t)^γ * log(p_t)
where γ = 2.5 (focusing parameter)
```

**Why Focal Loss for BFRB Detection:**
- **Class Imbalance:** BFRB instances are significantly rarer than normal activities
- **Hard Example Focus:** Emphasizes learning from difficult-to-classify borderline cases
- **Reduced False Positives:** Prevents model from being overwhelmed by easy negative examples

**Label Smoothing for Multiclass:**
```
L_smooth = (1-ε) * L_CE + ε/K
where ε = 0.15 (smoothing factor)
```

**Benefits:**
- **Overconfidence Prevention:** Reduces model overconfidence on training data
- **Better Calibration:** Improves probability estimates for uncertainty quantification

## Advanced Training Strategies

### 1. Mixup Data Augmentation

**Implementation:**
```
x_mixed = λ * x_i + (1-λ) * x_j
y_mixed = λ * y_i + (1-λ) * y_j
where λ ~ Beta(α=0.3, α=0.3)
```

**Why Mixup Works for Sensor Data:**
- **Interpolation Realism:** Mixed sensor readings represent plausible intermediate states
- **Regularization Effect:** Prevents overfitting to specific sensor patterns
- **Boundary Smoothing:** Creates smoother decision boundaries between gesture classes

### 2. Cosine Annealing with Warm Restarts

**Schedule Design:**
```
η_t = η_min + (η_max - η_min) * (1 + cos(π * T_cur/T_max)) / 2
```

**Benefits for BFRB Detection:**
- **Escape Local Minima:** Periodic restarts help escape suboptimal solutions
- **Multiple Convergence Attempts:** Each restart provides opportunity to find better solutions
- **Ensemble Effects:** Different restart phases can be ensemble for better performance

### 3. Cross-Validation Strategy

**Stratified K-Fold Implementation:**
- **Stratification:** Ensures balanced distribution of rare BFRB classes across folds
- **Temporal Consistency:** Maintains temporal relationships within sequences
- **Robust Evaluation:** 5-fold validation provides reliable performance estimates
