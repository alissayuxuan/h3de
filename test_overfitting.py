"""
Overfitting Test für VertebraePOI Dataset
==========================================
Testet ob das Model auf einem einzelnen Sample overfitting kann.
Ein funktionierendes Setup sollte Loss -> 0 erreichen.
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from data_utils.spine_dataloader import VertebraePOI
import data_utils.transforms as tr
from models.PBiFormer_Unet import PBiFormer_Unet
from models.losses import HNM_propmap


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'master_df_path': 'datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv',
    'input_shape': (215, 215, 144),
    'crop_size': [128, 128, 64],
    'n_epochs': 100,
    'learning_rate': 0.001,
    'n_samples': 1,  # Anzahl Samples für Overfitting
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'poi_indices': [
        81, 82, 83, 84, 85, 86, 87, 88, 89,
        101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116,
        117, 118, 119, 120, 121, 122, 123, 124,
        125, 127
    ]
}


# ============================================================================
# SETUP DATASET
# ============================================================================
print("="*80)
print("OVERFITTING TEST - VertebraePOI Dataset")
print("="*80)

# Load master dataframe
master_df = pd.read_csv(CONFIG['master_df_path'])
print(f"✓ Loaded master_df with {len(master_df)} total samples")

# Define transforms
train_transform = transforms.Compose([
    tr.RandomCrop(size=CONFIG['crop_size'], min_rate=0.6),
    tr.LandmarkProposal(
        size=CONFIG['crop_size'], 
        shrink=4, 
        anchors=[0.5, 0.75, 1., 1.25]
    ),
    tr.ToTensor(),
])

# Create full dataset
dataset_full = VertebraePOI(
    master_df=master_df,
    transform=train_transform,
    phase='train',
    input_data_type='ct',
    poi_indices=CONFIG['poi_indices'],
    input_shape=CONFIG['input_shape'],
    project_gt=False,  # Disable for faster testing
    show_neighbors=False
)

print(f"✓ Created dataset with {len(dataset_full)} training samples")

# Create mini dataset (only N samples)
dataset_mini = Subset(dataset_full, list(range(CONFIG['n_samples'])))
print(f"✓ Using {len(dataset_mini)} sample(s) for overfitting test")

# Create dataloader
loader_mini = DataLoader(
    dataset_mini, 
    batch_size=1, 
    shuffle=False,
    num_workers=0
)

# Inspect first sample
sample = dataset_full[0]
print(f"\n=== SAMPLE INSPECTION ===")
print(f"  Image shape:     {sample['image'].shape}")
print(f"  Image dtype:     {sample['image'].dtype}")
print(f"  Image range:     [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
print(f"  Proposals shape: {sample['proposals'].shape}")
print(f"  Spacing:         {sample['spacing']}")
print(f"  Filename:        {sample['filename']}")


# ============================================================================
# SETUP MODEL
# ============================================================================
print(f"\n=== MODEL SETUP ===")
device = torch.device(CONFIG['device'])
print(f"  Device: {device}")

# Initialize model
net = PBiFormer_Unet(n_class=35, n_anchor=4).to(device)
n_params = sum(p.numel() for p in net.parameters())
print(f"  Model parameters: {n_params:,}")

# Loss function
loss_fn = HNM_propmap(n_class=35, device=device).to(device)

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG['learning_rate'])
print(f"  Optimizer: Adam (lr={CONFIG['learning_rate']})")


# ============================================================================
# TRAINING LOOP
# ============================================================================
print(f"\n=== TRAINING ===")
print(f"Training for {CONFIG['n_epochs']} epochs on {CONFIG['n_samples']} sample(s)...")

net.train()
losses = []
epoch_losses = []

for epoch in range(CONFIG['n_epochs']):
    epoch_loss = 0.0
    n_batches = 0
    
    for sample in loader_mini:
        # Move to device
        data = sample['image'].to(device)
        proposals = sample['proposals'].to(device)
        
        # Forward pass
        proposal_map = net(data)
        
        # Compute loss
        loss = loss_fn(proposal_map, proposals)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record
        loss_value = loss.item()
        losses.append(loss_value)
        epoch_loss += loss_value
        n_batches += 1
    
    # Average epoch loss
    avg_epoch_loss = epoch_loss / n_batches
    epoch_losses.append(avg_epoch_loss)
    
    # Print progress
    if epoch % 10 == 0 or epoch == CONFIG['n_epochs'] - 1:
        print(f"  Epoch {epoch:3d}: Loss = {avg_epoch_loss:.6f}")


# ============================================================================
# RESULTS
# ============================================================================
print(f"\n=== RESULTS ===")
print(f"  Initial loss:    {losses[0]:.6f}")
print(f"  Final loss:      {losses[-1]:.6f}")
reduction = (1 - losses[-1]/losses[0]) * 100
print(f"  Loss reduction:  {reduction:.1f}%")

# Check if overfitting succeeded
if losses[-1] < losses[0] * 0.1:
    print(f"  ✓ SUCCESS: Loss decreased by >90% - model can overfit!")
    test_passed = True
else:
    print(f"  ✗ WARNING: Loss only decreased by {reduction:.1f}% - check setup!")
    test_passed = False

"""
# ============================================================================
# VISUALIZATION
# ============================================================================
print(f"\n=== VISUALIZATION ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: All iterations
axes[0].plot(losses, linewidth=1, alpha=0.7)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title(f'Training Loss (All Iterations)\nFinal: {losses[-1]:.6f}')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')  # Log scale für bessere Sichtbarkeit

# Plot 2: Per epoch
axes[1].plot(epoch_losses, linewidth=2, marker='o', markersize=3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Average Loss')
axes[1].set_title(f'Training Loss (Per Epoch)\nReduction: {reduction:.1f}%')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('overfit_test_vertebrae.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Plot saved to: overfit_test_vertebrae.png")

# Show plot
try:
    plt.show()
except:
    print("  (Could not display plot - running in headless mode)")
"""

# ============================================================================
# VISUALIZATION (ENHANCED)
# ============================================================================
print(f"\n=== VISUALIZATION ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: All iterations
axes[0, 0].plot(losses, linewidth=1, alpha=0.7)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title(f'Training Loss (All Iterations)\nFinal: {losses[-1]:.6f}')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Plot 2: Per epoch
axes[0, 1].plot(epoch_losses, linewidth=2, marker='o', markersize=3)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Average Loss')
axes[0, 1].set_title(f'Training Loss (Per Epoch)\nReduction: {reduction:.1f}%')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# Plot 3: Gradient norms
axes[1, 0].plot(grad_norms, linewidth=1, alpha=0.7, color='orange')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Gradient Norm')
axes[1, 0].set_title('Gradient Norms During Training')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# Plot 4: Loss decrease rate
loss_decrease = np.diff(epoch_losses)
axes[1, 1].plot(loss_decrease, linewidth=2, color='green')
axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss Decrease')
axes[1, 1].set_title('Loss Decrease Rate (should be negative)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfit_test_vertebrae_detailed.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Detailed plot saved to: overfit_test_vertebrae_detailed.png")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================
print(f"\n=== DETAILED ANALYSIS ===")

# Loss statistics
import numpy as np
losses_array = np.array(losses)

print(f"  Loss statistics:")
print(f"    Mean:     {losses_array.mean():.6f}")
print(f"    Std:      {losses_array.std():.6f}")
print(f"    Min:      {losses_array.min():.6f}")
print(f"    Max:      {losses_array.max():.6f}")
print(f"    Median:   {np.median(losses_array):.6f}")

# Convergence check
last_10_epochs = epoch_losses[-10:]
std_last_10 = np.std(last_10_epochs)
print(f"\n  Convergence:")
print(f"    Std (last 10 epochs): {std_last_10:.6f}")
if std_last_10 < 0.001:
    print(f"    ✓ Loss has converged (stable)")
else:
    print(f"    ⚠ Loss still changing (not converged)")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"Dataset:        {len(dataset_full)} samples (using {CONFIG['n_samples']} for test)")
print(f"Epochs:         {CONFIG['n_epochs']}")
print(f"Final loss:     {losses[-1]:.6f}")
print(f"Reduction:      {reduction:.1f}%")
print(f"Test result:    {'✓ PASSED' if test_passed else '✗ FAILED'}")
print("="*80)

# Exit code
if not test_passed:
    import sys
    print("\n⚠️  Test did not pass - investigate training setup!")
    sys.exit(1)
else:
    print("\n✅ Overfitting test successful - training setup works correctly!")