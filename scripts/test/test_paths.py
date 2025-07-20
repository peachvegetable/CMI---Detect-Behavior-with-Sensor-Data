#!/usr/bin/env python3
"""
Test script to verify all paths are working correctly after reorganization
"""

import os
import sys

# Get the root directory of the project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

print("Path Resolution Test")
print("=" * 60)
print(f"Script directory: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print()

# Test file existence
files_to_check = [
    ('data/train.csv', os.path.join(DATA_DIR, 'train.csv')),
    ('data/test.csv', os.path.join(DATA_DIR, 'test.csv')),
    ('scripts/model.py', os.path.join(PROJECT_ROOT, 'scripts', 'model.py')),
    ('scripts/train/train_improved.py', os.path.join(PROJECT_ROOT, 'scripts', 'train', 'train_improved.py')),
    ('improved_model_fold_0.pth', os.path.join(PROJECT_ROOT, 'improved_model_fold_0.pth')),
]

print("File existence check:")
print("-" * 60)
all_exist = True
for name, path in files_to_check:
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {name}: {exists}")
    if not exists:
        all_exist = False

print()
if all_exist:
    print("✓ All paths are correctly configured!")
else:
    print("✗ Some paths need attention")

# Test import paths
print("\nTesting Python imports...")
print("-" * 60)

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

try:
    import model
    print("✓ Successfully imported model.py")
except ImportError as e:
    print(f"✗ Failed to import model.py: {e}")

print("\n✓ Path test complete!")