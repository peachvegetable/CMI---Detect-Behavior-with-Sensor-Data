import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Test the attention mask creation
def test_attention_mask():
    print("Testing Attention Mask Implementation")
    print("="*50)
    
    # Simulate a batch with different sequence lengths
    batch_size = 3
    max_seq_len = 100
    n_features = 344  # typical feature count
    
    # Create dummy data
    sequences = torch.randn(batch_size, n_features, max_seq_len)
    
    # Original lengths before padding
    original_lengths = torch.tensor([50, 75, 100])
    
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Original sequence lengths: {original_lengths.tolist()}")
    
    # Test mask creation
    seq_range = torch.arange(0, max_seq_len)
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_seq_len)
    
    lengths_expanded = original_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
    mask = seq_range < lengths_expanded
    
    print("\nAttention mask shape:", mask.shape)
    print("\nMask for each sequence (showing first 10 and last 10 positions):")
    
    for i in range(batch_size):
        mask_seq = mask[i]
        print(f"\nSequence {i+1} (length={original_lengths[i]}):")
        print(f"  First 10: {mask_seq[:10].int().tolist()}")
        print(f"  Last 10:  {mask_seq[-10:].int().tolist()}")
        print(f"  Sum of True values: {mask_seq.sum().item()} (should equal {original_lengths[i].item()})")
    
    # Test masking attention weights
    print("\n" + "="*50)
    print("Testing attention weight masking:")
    
    # Simulate attention logits
    attention_logits = torch.randn(batch_size, max_seq_len)
    
    # Apply mask
    masked_logits = attention_logits.masked_fill(~mask, float('-inf'))
    
    # Apply softmax
    attention_weights = F.softmax(masked_logits, dim=1)
    
    print("\nAttention weights sum for each sequence:")
    for i in range(batch_size):
        weights_sum = attention_weights[i].sum().item()
        print(f"  Sequence {i+1}: {weights_sum:.6f} (should be ~1.0)")
        
        # Check that padded positions have ~0 weight
        padded_weights = attention_weights[i, original_lengths[i]:].sum().item()
        print(f"    Sum of padded position weights: {padded_weights:.10f} (should be ~0)")
    
    # Test with downsampled lengths (as in the model)
    print("\n" + "="*50)
    print("Testing with downsampled lengths (stride=2):")
    
    downsampled_lengths = (original_lengths + 1) // 2
    downsampled_seq_len = (max_seq_len + 1) // 2
    
    print(f"Downsampled lengths: {downsampled_lengths.tolist()}")
    print(f"Downsampled sequence length: {downsampled_seq_len}")
    
    # Create mask for downsampled sequence
    seq_range_ds = torch.arange(0, downsampled_seq_len)
    seq_range_ds = seq_range_ds.unsqueeze(0).expand(batch_size, downsampled_seq_len)
    
    lengths_expanded_ds = downsampled_lengths.unsqueeze(1).expand(batch_size, downsampled_seq_len)
    mask_ds = seq_range_ds < lengths_expanded_ds
    
    print("\nDownsampled mask verification:")
    for i in range(batch_size):
        true_count = mask_ds[i].sum().item()
        expected = downsampled_lengths[i].item()
        print(f"  Sequence {i+1}: {true_count} True values (expected: {expected})")

if __name__ == "__main__":
    test_attention_mask()
    print("\n✓ All tests completed!")