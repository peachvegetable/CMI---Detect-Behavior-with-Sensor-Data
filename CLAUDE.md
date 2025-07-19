# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition project for "Child Mind Institute - Detect Behavior with Sensor Data". The goal is to develop predictive models that distinguish body-focused repetitive behaviors (BFRBs) like hair pulling from non-BFRB everyday gestures using multimodal sensor data from a wrist-worn device called Helios.

## Core Commands

```bash
# Main training script
python train.py

# Data exploration and visualization
python data_exploration/acce_examine.py

# Run Jupyter notebook for detailed analysis
jupyter lab data_exploration/data_exploration_detailed.ipynb
```

## Model Architecture

The repository contains neural network architectures in `model.py`:

- **CompetitionModel**: Main multi-task CNN+LSTM with attention mechanism
  - Handles both IMU-only and full sensor scenarios
  - Multi-task learning: binary (BFRB vs non-BFRB) + multiclass classification
  - Sensor-specific feature processors for IMU, thermopile, and time-of-flight data

- **EnhancedCNNLSTM**: CNN-LSTM with attention and residual connections
- **CNNLSTM**: Basic CNN-LSTM baseline model

## Training Process (train.py)

Key training configuration:
- **Window size**: 30 timesteps with stride 10 for data augmentation
- **Behavior filtering**: Focus on "Performs gesture" phase only
- **Multi-task loss**: Combines binary (60%) and multiclass (40%) objectives
- **Optimizer**: AdamW with OneCycleLR scheduler
- **Early stopping**: Based on competition score (average of binary F1 and macro F1)

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

## Kaggle Evaluation

```bash
# Test inference server locally
cd data/kaggle_evaluation
python cmi_inference_server.py
```

The evaluation API serves test data one sequence at a time and expects gesture predictions.

## Development Environment

- **Conda Environment:**
  - Always use conda env cmi for this codebase

## Motivation and Guiding Principles

- **Project Philosophy**:
  - give it your all, don't hold back