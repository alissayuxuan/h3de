"""
Overfitting Test für H3DE-Net Training Pipeline
================================================
Testet ob das komplette Training-Setup funktioniert, indem es
auf 1-5 Samples overfittet.

Usage:
    python test_overfitting.py --n_samples 1 --epochs 500
"""

import argparse
import os
import time
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from data_utils.spine_dataloader import VertebraePOI
import data_utils.transforms as tr
from utils import setgpu
from models.losses import HNM_propmap
from models.PBiFormer_Unet import PBiFormer_Unet


# ============================================================================
# ARGUMENT PARSER
# ============================================================================
parser = argparse.ArgumentParser(description='Overfitting Test for H3DE-Net')

# Test-specific args
parser.add_argument('--n_samples', default=1, type=int, 
                    help='Number of samples to overfit on')
parser.add_argument('--epochs', default=500, type=int, 
                    help='Number of training epochs')
parser.add_argument('--lr', default=0.01, type=float,
                    help='Learning rate (higher for overfitting)')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size (keep at 1 for overfitting)')

# Model args
parser.add_argument('--model_name', default='PBiFormer_Unet', type=str)
parser.add_argument('--n_class', default=35, type=int)
parser.add_argument('--shrink', default=4, type=int)
parser.add_argument('--anchors', default=[0.5, 0.75, 1., 1.25], type=list)

# Data args
parser.add_argument('--master_df_path', 
                    default='datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv',
                    type=str)
parser.add_argument('--input_shape', default=[215, 215, 144], type=int, nargs=3)
parser.add_argument('--crop_size', default=[128, 128, 64], type=int, nargs=3)

# Training args
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='Weight decay (0 for overfitting)')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--save_dir', default='./overfitting_test_results', type=str)

# Debugging args
parser.add_argument('--disable_crop', action='store_true',
                    help='Disable random crop for faster overfitting')
parser.add_argument('--project_gt', action='store_true',
                    help='Enable ground truth surface projection')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# SETUP LOGGING
# ============================================================================
def setup_logging(save_dir):
    """Setup logging to both file and console"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'overfitting_test.log')
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_file


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_epoch(data_loader, net, loss_fn, optimizer, epoch):
    """Train for one epoch"""
    net.train()
    epoch_losses = []
    grad_norms = []
    
    for i, sample in enumerate(data_loader):
        data = sample['image'].to(DEVICE)
        proposals = sample['proposals'].to(DEVICE)
        
        # Forward
        proposal_map = net(data)
        
        # Loss
        cur_loss = loss_fn(proposal_map, proposals)
        
        # Backward
        optimizer.zero_grad()
        cur_loss.backward()
        
        # Track gradient norm
        total_norm = 0
        for p in net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        # Optimize
        optimizer.step()
        
        epoch_losses.append(cur_loss.item())
    
    avg_loss = np.mean(epoch_losses)
    avg_grad = np.mean(grad_norms)
    
    return avg_loss, avg_grad


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(losses, grad_norms, save_dir):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = np.arange(1, len(losses) + 1)
    
    # Plot 1: Loss over epochs (log scale)
    axes[0, 0].plot(epochs, losses, linewidth=2, marker='o', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training Loss\nFinal: {losses[-1]:.6f}')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Loss reduction
    reduction = (1 - np.array(losses) / losses[0]) * 100
    axes[0, 1].plot(epochs, reduction, linewidth=2, color='green')
    axes[0, 1].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% target')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss Reduction (%)')
    axes[0, 1].set_title(f'Loss Reduction\nFinal: {reduction[-1]:.1f}%')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Gradient norms
    axes[1, 0].plot(epochs, grad_norms, linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Norms')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Loss decrease rate
    loss_decrease = -np.diff(losses)
    axes[1, 1].plot(epochs[1:], loss_decrease, linewidth=2, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Decrease')
    axes[1, 1].set_title('Loss Decrease Rate (should be positive)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'overfitting_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logging.info(f"Plot saved to: {plot_path}")
    
    return plot_path


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================
def main(args):
    logging.info("="*80)
    logging.info("OVERFITTING TEST")
    logging.info("="*80)
    logging.info(f"Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    logging.info("="*80)
    
    # Setup
    cudnn.benchmark = True
    setgpu(args.gpu)
    
    # ===== MODEL =====
    logging.info("\n=== MODEL SETUP ===")
    net = PBiFormer_Unet(n_class=args.n_class, n_anchor=len(args.anchors))
    net = net.to(DEVICE)
    
    n_params = sum(p.numel() for p in net.parameters())
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Parameters: {n_params:,}")
    logging.info(f"Device: {DEVICE}")
    
    if len(args.gpu.split(',')) > 1:
        net = DataParallel(net)
        logging.info(f"Using DataParallel with GPUs: {args.gpu}")
    
    # ===== LOSS =====
    loss_fn = HNM_propmap(n_class=args.n_class, device=DEVICE).to(DEVICE)
    
    # ===== OPTIMIZER =====
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay
    )
    logging.info(f"Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    
    # ===== DATA =====
    logging.info("\n=== DATA SETUP ===")
    master_df = pd.read_csv(args.master_df_path)
    logging.info(f"Loaded master_df: {len(master_df)} samples")
    
    # Define transforms
    if args.disable_crop:
        logging.info("Random crop DISABLED for faster overfitting")
        train_transform = transforms.Compose([
            tr.LandmarkProposal(
                size=args.crop_size,
                shrink=args.shrink, 
                anchors=args.anchors
            ),
            tr.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            tr.RandomCrop(size=args.crop_size, min_rate=0.6),
            tr.LandmarkProposal(
                size=args.crop_size,
                shrink=args.shrink, 
                anchors=args.anchors
            ),
            tr.ToTensor(),
        ])
    
    # Create full dataset
    full_dataset = VertebraePOI(
        master_df=master_df,
        transform=train_transform,
        phase='train',
        input_shape=args.input_shape,
        project_gt=args.project_gt
    )
    
    logging.info(f"Full dataset size: {len(full_dataset)}")
    
    # Create subset
    n_samples = min(args.n_samples, len(full_dataset))
    train_dataset = Subset(full_dataset, list(range(n_samples)))
    logging.info(f"Using {n_samples} sample(s) for overfitting test")
    
    # DataLoader
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffle for reproducibility
        num_workers=0   # No multiprocessing for debugging
    )
    
    # Inspect first sample
    sample = full_dataset[0]
    logging.info(f"\nSample inspection:")
    logging.info(f"  Filename: {sample['filename']}")
    logging.info(f"  Image shape: {sample['image'].shape}")
    logging.info(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    logging.info(f"  Proposals shape: {sample['proposals'].shape}")
    logging.info(f"  Spacing: {sample['spacing']}")
    
    landmarks = sample['landmarks']
    invalid_mask = np.all(landmarks == -1000, axis=1)
    n_invalid = invalid_mask.sum()
    logging.info(f"Invalid POIs: {n_invalid}/35")

    # ===== INITIAL FORWARD PASS =====
    logging.info("\n=== INITIAL FORWARD PASS ===")
    net.eval()
    with torch.no_grad():
        sample_batch = next(iter(trainloader))
        data = sample_batch['image'].to(DEVICE)
        proposals = sample_batch['proposals'].to(DEVICE)
        
        output = net(data)
        initial_loss = loss_fn(output, proposals)
        
        logging.info(f"Input shape: {data.shape}")
        logging.info(f"Output shape: {output.shape}")
        logging.info(f"Initial loss: {initial_loss.item():.6f}")
    
    # ===== TRAINING LOOP =====
    logging.info("\n=== TRAINING ===")
    logging.info(f"Training for {args.epochs} epochs...")
    
    epoch_losses = []
    epoch_grads = []
    
    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_grad = train_epoch(trainloader, net, loss_fn, optimizer, epoch)
        epoch_losses.append(avg_loss)
        epoch_grads.append(avg_grad)
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            logging.info(f"Epoch {epoch:3d}/{args.epochs}: Loss = {avg_loss:.6f}, Grad = {avg_grad:.6f}")
    
    # ===== RESULTS =====
    logging.info("\n" + "="*80)
    logging.info("RESULTS")
    logging.info("="*80)
    
    initial_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]
    reduction = (1 - final_loss / initial_loss) * 100
    
    logging.info(f"Initial loss:    {initial_loss:.6f}")
    logging.info(f"Final loss:      {final_loss:.6f}")
    logging.info(f"Loss reduction:  {reduction:.1f}%")
    
    # Check convergence
    last_10_losses = epoch_losses[-10:]
    std_last_10 = np.std(last_10_losses)
    logging.info(f"Std (last 10):   {std_last_10:.6f}")
    
    # Gradient analysis
    logging.info(f"\nGradient norms:")
    logging.info(f"  Mean:    {np.mean(epoch_grads):.6f}")
    logging.info(f"  Final:   {epoch_grads[-1]:.6f}")
    
    if np.mean(epoch_grads) < 1e-6:
        logging.warning("⚠️  Very small gradients - vanishing gradient problem!")
    if np.mean(epoch_grads) > 100:
        logging.warning("⚠️  Very large gradients - exploding gradient problem!")
    
    # Pass/Fail
    test_passed = reduction >= 90.0
    
    if test_passed:
        logging.info("\n✅ SUCCESS: Loss decreased by ≥90% - model can overfit!")
    else:
        logging.warning(f"\n⚠️  WARNING: Loss only decreased by {reduction:.1f}%")
        logging.warning("Possible issues:")
        logging.warning("  1. Learning rate too low (try --lr 0.01)")
        logging.warning("  2. Not enough epochs (try --epochs 500)")
        logging.warning("  3. Random crop prevents overfitting (try --disable_crop)")
        logging.warning("  4. Weight decay too high (should be 0 for overfitting)")
    
    # ===== VISUALIZATION =====
    logging.info("\n=== VISUALIZATION ===")
    plot_path = plot_results(epoch_losses, epoch_grads, args.save_dir)
    
    # ===== SAVE RESULTS =====
    results = {
        'args': vars(args),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'reduction': reduction,
        'epoch_losses': epoch_losses,
        'epoch_grads': epoch_grads,
        'test_passed': test_passed
    }
    
    results_path = os.path.join(args.save_dir, 'results.pth')
    torch.save(results, results_path)
    logging.info(f"Results saved to: {results_path}")
    
    # ===== SUMMARY =====
    logging.info("\n" + "="*80)
    logging.info("SUMMARY")
    logging.info("="*80)
    logging.info(f"Dataset:         {len(full_dataset)} total ({n_samples} used)")
    logging.info(f"Epochs trained:  {args.epochs}")
    logging.info(f"Final loss:      {final_loss:.6f}")
    logging.info(f"Loss reduction:  {reduction:.1f}%")
    logging.info(f"Test result:     {'✅ PASSED' if test_passed else '⚠️ FAILED'}")
    logging.info(f"Plot:            {plot_path}")
    logging.info(f"Results:         {results_path}")
    logging.info("="*80)
    
    return test_passed


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.save_dir)
    
    print("\n" + "="*80)
    print("OVERFITTING TEST - H3DE-Net Training Pipeline")
    print("="*80)
    print(f"Log file: {log_file}")
    print(f"Samples:  {args.n_samples}")
    print(f"Epochs:   {args.epochs}")
    print(f"LR:       {args.lr}")
    print(f"GPU:      {args.gpu}")
    print("="*80 + "\n")
    
    # Run test
    test_passed = main(args)
    
    # Exit code
    import sys
    if not test_passed:
        sys.exit(1)
    else:
        sys.exit(0)