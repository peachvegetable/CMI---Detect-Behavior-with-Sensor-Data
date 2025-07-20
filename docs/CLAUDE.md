# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition project for "Child Mind Institute - Detect Behavior with Sensor Data". The goal is to develop predictive models that distinguish body-focused repetitive behaviors (BFRBs) like hair pulling from non-BFRB everyday gestures using multimodal sensor data from a wrist-worn device called Helios.

## Repository Structure

```
CMI---Detect-Behavior-with-Sensor-Data/
├── data/                          # Competition data and evaluation API
│   ├── train.csv, test.csv      # Main datasets
│   └── kaggle_evaluation/        # Kaggle submission API
├── data_exploration/             # Data analysis and visualization
│   ├── acce_examine.py          # Acceleration plot generator
│   └── data_exploration_detailed.ipynb
├── docs/                         # Documentation
│   ├── CLAUDE.md                # This file
│   ├── model_explanation_qa.md  # Model architecture Q&A
│   └── competition_summary.md   # Competition overview
├── scripts/                      # Main codebase
│   ├── model.py                 # Model architectures
│   ├── inference_improved.py    # Inference pipeline
│   ├── train/                   # Training scripts
│   │   ├── train_best.py       # Best performing training script
│   │   └── train_improved.py   # Improved training with attention masking
│   ├── test/                    # Test scripts
│   │   ├── test_best.py        # Model testing
│   │   └── test_attention_mask.py
│   └── submission/              # Submission scripts
│       ├── submission.py        # Basic submission
│       └── kaggle_submission_improved.py
├── accel_plots_by_orientation/   # Gesture visualization plots
└── improved_model_fold_*.pth     # Trained model checkpoints
```

## Core Commands

**Important**: Run all commands from the project root directory.

```bash
# Navigate to project root
cd /path/to/CMI---Detect-Behavior-with-Sensor-Data

# Main training scripts
python scripts/train/train_best.py           # Best performing model
python scripts/train/train_improved.py       # With attention masking

# Model inference
python scripts/inference_improved.py

# Data exploration and visualization
python data_exploration/acce_examine.py

# Run Jupyter notebook for detailed analysis
jupyter lab data_exploration/data_exploration_detailed.ipynb

# Test model performance
python scripts/test/test_best.py

# Generate Kaggle submission
python scripts/submission/kaggle_submission_improved.py

# Test path configuration
python scripts/test/test_paths.py
```

## Model Architecture

The repository contains neural network architectures in `scripts/model.py`:

- **ImprovedBFRBModel**: Latest model with attention masking support
  - Dynamic feature engineering (angular velocity, jerk, MAD)
  - Separate encoders for IMU, thermopile, and TOF data
  - Attention mechanism with proper padding handling
  - Multi-task outputs: binary + multiclass classification

- **CompetitionModel**: Original multi-task CNN+LSTM with attention
  - Handles both IMU-only and full sensor scenarios
  - Multi-task learning: binary (BFRB vs non-BFRB) + multiclass classification

## Training Process

### Latest Training Script (`scripts/train/train_improved.py`)

Key improvements:
- **Attention masking**: Properly handles variable-length sequences
- **Dynamic padding**: Pads only to max length in each batch
- **Advanced feature engineering**:
  - Angular velocity from quaternions
  - Angular distance traveled
  - Jerk (acceleration derivative)
  - Mean Absolute Deviation (MAD)
  - Acceleration magnitude
- **No behavior filtering**: Works with full dataset structure
- **Robust data handling**: Handles NaN/inf values gracefully

Configuration:
- **Batch size**: 32
- **Learning rate**: 1e-4 with CosineAnnealingWarmRestarts
- **Multi-task loss**: 60% binary + 40% multiclass
- **Label smoothing**: 0.15
- **Focal loss gamma**: 2.5
- **Mixup augmentation**: α = 0.3
- **Early stopping patience**: 20 epochs

## Data Structure

**Sensor Data (332 features per timestep):**
- IMU: acceleration (3) + rotation quaternion (4) = 7 channels
- Thermopiles: 5 temperature sensors = 5 channels  
- Time-of-Flight: 5 sensors × 64 pixels = 320 channels

**BFRB Gestures (8 target behaviors):**
- Above ear - pull hair, Forehead - pull hairline, Forehead - scratch
- Eyebrow - pull hair, Eyelash - pull hair, Neck - pinch skin
- Neck - scratch, Cheek - pinch skin

**Data Processing:**
- Sensor-specific RobustScaler normalization
- TOF missing data (-1) preserved as "no object" signal
- Thermopile missing data filled with median values

## Competition Constraints

- **Evaluation**: Average of binary F1 and macro F1 scores
- **Test Set Split**: Half contains IMU-only data (other sensors nulled)
- **Submission**: Must use Kaggle evaluation API in `data/kaggle_evaluation/`

## Data Exploration

The `accel_plots_by_orientation/` directory contains acceleration plots for different gestures and body orientations. Use `data_exploration/acce_examine.py` to generate new visualizations.

Key insights documented in:
- `data_exploration/key_movement_insights.txt`: Movement pattern analysis
- `data_exploration/competition_strategy_and_next_steps.txt`: Strategic recommendations

## Kaggle Evaluation

```bash
# Test inference server locally
cd data/kaggle_evaluation
python cmi_inference_server.py

# Run submission script
python scripts/submission/kaggle_submission_improved.py
```

The evaluation API serves test data one sequence at a time and expects gesture predictions.

## Feature Engineering Details

### Angular Velocity Calculation
- Computed from quaternion differences between consecutive timesteps
- Assumes 50Hz sampling rate (0.02s between samples)
- See `calculate_angular_velocity()` in training scripts

### Statistical Features
- **MAD (Mean Absolute Deviation)**: Rolling window statistics
- **Jerk**: Derivative of acceleration
- **Angular Distance**: Cumulative rotation traveled
- **Acceleration Magnitude**: L2 norm of acc_x, acc_y, acc_z

## Development Environment

- **Conda Environment:**
  - Always use conda env `cmi` for this codebase
  - Python 3.10+ recommended

## Recent Improvements

### Attention Masking (Latest)
- Model now properly handles variable-length sequences
- Padding positions are masked during attention computation
- Global pooling only aggregates over real data, not padding
- See `test_attention_mask.py` for validation

### Model Checkpoints
- Best models saved as `improved_model_fold_0.pth` through `fold_4.pth`
- Each checkpoint includes:
  - Model state dict
  - Scaler for normalization
  - Label encoder
  - Configuration used
  - Feature columns

## Motivation and Guiding Principles

- **Project Philosophy**:
  - Give it your all, don't hold back
  - Focus on robust, generalizable solutions
  - Properly handle edge cases (padding, missing data)
  - Document insights and reasoning