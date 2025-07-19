#!/usr/bin/env python3
"""
Check the scores stored in the model checkpoints
"""

import torch
import os

# Check all available model files
model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
print(f"Found {len(model_files)} model files:\n")

for model_file in sorted(model_files):
    try:
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        
        print(f"\n{model_file}:")
        if 'score' in checkpoint:
            print(f"  Competition Score: {checkpoint['score']:.4f}")
        if 'config' in checkpoint:
            print(f"  Max Epochs: {checkpoint['config'].get('max_epochs', 'N/A')}")
            print(f"  Window Size: {checkpoint['config'].get('max_sequence_length', checkpoint['config'].get('window_size', 'N/A'))}")
        if 'epoch' in checkpoint:
            print(f"  Best Epoch: {checkpoint['epoch'] + 1}")
        
    except Exception as e:
        print(f"\n{model_file}: Error loading - {str(e)}")