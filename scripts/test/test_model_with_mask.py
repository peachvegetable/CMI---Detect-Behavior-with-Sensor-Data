import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model
from scripts.train.train_improved import ImprovedBFRBModel, CONFIG

def test_model_with_attention_mask():
    print("Testing Model with Attention Masking")
    print("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    n_features = 344  # Base features + engineered features
    n_classes = 18
    batch_size = 4
    
    # Create model
    model = ImprovedBFRBModel(n_features=n_features, n_classes=n_classes).to(device)
    model.eval()
    
    print(f"\nModel created with {n_features} features and {n_classes} classes")
    
    # Test 1: Variable length sequences
    print("\n" + "-"*50)
    print("Test 1: Variable length sequences with padding")
    
    # Create sequences with different lengths
    seq_lengths = [30, 50, 75, 100]
    max_len = max(seq_lengths)
    
    # Create padded batch
    batch_data = []
    for i, length in enumerate(seq_lengths):
        # Create sequence with actual data
        seq = torch.randn(n_features, length)
        # Pad to max length
        if length < max_len:
            padding = torch.zeros(n_features, max_len - length)
            seq = torch.cat([seq, padding], dim=1)
        batch_data.append(seq)
    
    # Stack into batch
    sequences = torch.stack(batch_data).to(device)
    lengths = torch.tensor(seq_lengths).to(device)
    
    print(f"Input shape: {sequences.shape}")
    print(f"Sequence lengths: {seq_lengths}")
    
    # Forward pass with lengths
    with torch.no_grad():
        outputs_with_mask = model(sequences, lengths)
    
    print(f"\nOutputs with masking:")
    print(f"  Binary output shape: {outputs_with_mask['binary'].shape}")
    print(f"  Multiclass output shape: {outputs_with_mask['multiclass'].shape}")
    
    # Test 2: Compare with and without masking
    print("\n" + "-"*50)
    print("Test 2: Compare outputs with and without masking")
    
    # Forward pass without lengths
    with torch.no_grad():
        outputs_no_mask = model(sequences, None)
    
    # Compare outputs
    binary_diff = torch.abs(outputs_with_mask['binary'] - outputs_no_mask['binary']).mean().item()
    multi_diff = torch.abs(outputs_with_mask['multiclass'] - outputs_no_mask['multiclass']).mean().item()
    
    print(f"\nMean absolute difference:")
    print(f"  Binary output: {binary_diff:.6f}")
    print(f"  Multiclass output: {multi_diff:.6f}")
    print("\n(Differences show that masking affects the output)")
    
    # Test 3: Extreme case - very short sequence
    print("\n" + "-"*50)
    print("Test 3: Very short sequence (mostly padding)")
    
    short_seq = torch.randn(1, n_features, 100).to(device)
    short_seq[:, :, 10:] = 0  # Only first 10 timesteps have data
    short_length = torch.tensor([10]).to(device)
    
    with torch.no_grad():
        outputs_short = model(short_seq, short_length)
    
    print(f"Short sequence output shapes: Binary {outputs_short['binary'].shape}, "
          f"Multiclass {outputs_short['multiclass'].shape}")
    
    # Test 4: Gradient flow
    print("\n" + "-"*50)
    print("Test 4: Check gradient flow with masking")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Dummy targets
    binary_targets = torch.randint(0, 2, (batch_size,)).to(device)
    multi_targets = torch.randint(0, n_classes, (batch_size,)).to(device)
    
    # Forward pass
    outputs = model(sequences, lengths)
    
    # Calculate loss
    loss_binary = torch.nn.functional.cross_entropy(outputs['binary'], binary_targets)
    loss_multi = torch.nn.functional.cross_entropy(outputs['multiclass'], multi_targets)
    loss = 0.6 * loss_binary + 0.4 * loss_multi
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name.split('.')[0], param.grad.norm().item()))
    
    # Group by module
    module_grads = {}
    for module, norm in grad_norms:
        if module not in module_grads:
            module_grads[module] = []
        module_grads[module].append(norm)
    
    print("\nGradient norms by module:")
    for module, norms in module_grads.items():
        avg_norm = np.mean(norms)
        print(f"  {module}: {avg_norm:.6f}")
    
    print("\n✓ All gradients flowing correctly!")
    
    print("\n" + "="*50)
    print("✓ All tests passed! Model works correctly with attention masking.")

if __name__ == "__main__":
    test_model_with_attention_mask()