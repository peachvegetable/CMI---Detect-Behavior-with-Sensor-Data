# Scripts Directory Structure

## Main Scripts

### Training
- `train/train_improved.py` - Main training script with all fixes applied:
  - Dual scaling (IMU vs ToF/Thermal)
  - New features (rhythm signature, angular velocity magnitude)
  - StratifiedGroupKFold cross-validation
  - Dual evaluation for realistic CV scores

### Submission
- `submission/kaggle_submission_improved.py` - Kaggle submission script with matching preprocessing

### Utilities
- `check_model_scores.py` - Check scores in saved model checkpoints
- `inference_improved.py` - Local inference script for testing
- `utils/preprocessing.py` - Preprocessing utilities

### Legacy/Reference
- `train/train_best.py` - Older training script (kept for reference)
- `model.py` - Standalone model architecture
- `submission/submission.py` - Original submission script

## Model Files
All trained models and results have been moved to `/models/` directory.

## Removed Files
The following temporary debugging scripts were removed:
- analyze_remaining_gap.py
- test_different_imu_ratios.py  
- quick_fixes.py
- dual_pathway_model.py
- Various test scripts for debugging