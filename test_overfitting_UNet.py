import argparse
import os
import time
import numpy as np
import pandas as pd
import logging
import sys

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from data_utils.spine_dataloader import VertebraePOI
import data_utils.transforms as tr
from utils import setgpu
from data_utils.transforms import LandMarkToGaussianHeatMap
from models.losses import HNM_heatmap
from models.UNet import UNet3D

"""
Overfitting Test Script for UNet3D
===================================
Tests if the model can overfit to a small subset of data.
If the model can't overfit, there's likely a bug in the model or training setup.
"""

parser = argparse.ArgumentParser(description='UNet3D Overfitting Test')
parser.add_argument('--model_name', default='UNet3D', type=str)
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to overfit')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight-decay', default=0.0005, type=float)
parser.add_argument('--save_dir', default='./overfit_test/overfit_test_unet', type=str)
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--loss_name', default='HNM_heatmap', type=str)
parser.add_argument('--data_path', default='./gruber_dataset_cutouts', type=str)
parser.add_argument('--n_class', default=35, type=int)
parser.add_argument('--focus_radius', default=20, type=int)
parser.add_argument('--master_df_path', default='./datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv', type=str)
parser.add_argument('--n_samples', default=5, type=int, help='Number of samples to overfit on')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(save_dir):
    """Setup logging to both file and console"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = os.path.join(save_dir, 'overfit_log.txt')
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_file

def train_epoch(dataloader, net, loss_fn, optimizer, l2h, epoch):
    """Train for one epoch"""
    net.train()
    total_loss = []
    
    for i, sample in enumerate(dataloader):
        data = sample['image']
        landmark = sample['landmarks']
        
        # Generate heatmap
        heatmap_batch = l2h(landmark)
        data = data.to(DEVICE)
        
        # Forward pass
        heatmap = net(data)
        
        # Compute loss
        optimizer.zero_grad()
        cur_loss = loss_fn(heatmap, heatmap_batch)
        total_loss.append(cur_loss.item())
        
        # Backward pass
        cur_loss.backward()
        optimizer.step()
    
    mean_loss = np.mean(total_loss)
    return mean_loss

def eval_epoch(dataloader, net, loss_fn, l2h, epoch):
    """Evaluate for one epoch"""
    net.eval()
    total_loss = []
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            data = sample['image']
            landmark = sample['landmarks']
            
            heatmap_batch = l2h(landmark)
            data = data.to(DEVICE)
            
            heatmap = net(data)
            cur_loss = loss_fn(heatmap, heatmap_batch)
            total_loss.append(cur_loss.item())
    
    mean_loss = np.mean(total_loss)
    return mean_loss

def main(args):
    cudnn.benchmark = True
    setgpu(args.gpu)
    
    # Setup logging
    log_file = setup_logging(args.save_dir)
    
    print("\n" + "="*80)
    print("UNet3D OVERFITTING TEST")
    print("="*80)
    logging.info("="*80)
    logging.info("UNet3D Overfitting Test Started")
    logging.info("="*80)
    logging.info(f"Configuration: {args}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Device: {DEVICE}")
    
    # Initialize model
    logging.info("\n1. Initializing UNet3D model...")
    net = UNet3D(n_class=args.n_class)
    net = net.to(DEVICE)
    
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f"   Total parameters: {total_params:,}")
    logging.info(f"   Trainable parameters: {trainable_params:,}")
    
    # Initialize loss
    loss_fn = HNM_heatmap(R=args.focus_radius)
    loss_fn = loss_fn.to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay
    )
    
    # Generate Gaussian Heatmap converter
    l2h = LandMarkToGaussianHeatMap(
        R=args.focus_radius, 
        n_class=args.n_class,
        GPU=DEVICE, 
        img_size=(128, 128, 64)
    )
    
    # Load dataset
    logging.info(f"\n2. Loading {args.n_samples} training samples for overfitting test...")
    master_df = pd.read_csv(args.master_df_path)
    
    train_transform = transforms.Compose([
        tr.RandomCrop(size=[128, 128, 64]),
        tr.ToTensor(),
    ])
    
    full_dataset = VertebraePOI(
        master_df=master_df,
        transform=train_transform,
        phase='train',
        input_shape=(215, 215, 144),
        project_gt=True
    )
    
    # Create small subset for overfitting
    indices = list(range(min(args.n_samples, len(full_dataset))))
    train_dataset = Subset(full_dataset, indices)
    
    logging.info(f"   Using {len(train_dataset)} samples from full dataset ({len(full_dataset)} total)")
    
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Evaluation loader (same data, no shuffle)
    evalloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    logging.info(f"   Batches per epoch: {len(trainloader)}")
    
    # Training loop
    logging.info(f"\n3. Starting overfitting test for {args.epochs} epochs...")
    logging.info("="*80)
    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(trainloader, net, loss_fn, optimizer, l2h, epoch)
        
        # Evaluate (on same data)
        eval_loss = eval_epoch(evalloader, net, loss_fn, l2h, epoch)
        
        losses.append(train_loss)
        epoch_time = time.time() - start_time
        
        # Log every epoch
        logging.info(
            f'Epoch [{epoch:3d}/{args.epochs}] | '
            f'Train Loss: {train_loss:.6f} | '
            f'Eval Loss: {eval_loss:.6f} | '
            f'Time: {epoch_time:.1f}s'
        )
        
        # Save best model
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': eval_loss,
            }, os.path.join(args.save_dir, 'best_overfit_model.ckpt'))
        
        # Check for overfitting success
        if epoch >= 10 and train_loss < 0.01:
            logging.info("\n" + "="*80)
            logging.info("🎉 SUCCESS! Model successfully overfitted to training data!")
            logging.info(f"   Final train loss: {train_loss:.6f}")
            logging.info(f"   Reached at epoch: {epoch}")
            logging.info("="*80)
            break
    
    # Final summary
    logging.info("\n" + "="*80)
    logging.info("OVERFITTING TEST SUMMARY")
    logging.info("="*80)
    logging.info(f"Final train loss: {train_loss:.6f}")
    logging.info(f"Final eval loss: {eval_loss:.6f}")
    logging.info(f"Best eval loss: {best_loss:.6f}")
    logging.info(f"Total epochs run: {epoch}")
    
    # Loss reduction analysis
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    logging.info(f"\nLoss reduction: {initial_loss:.6f} → {final_loss:.6f} ({reduction:.1f}%)")
    
    if final_loss < 0.1:
        logging.info("\n✓ PASS: Model can overfit (loss < 0.1)")
        logging.info("  The model architecture and training setup are working correctly.")
    elif final_loss < 0.5:
        logging.info("\n⚠ PARTIAL: Model is learning but slowly (loss < 0.5)")
        logging.info("  Consider: higher learning rate, more epochs, or check data augmentation.")
    else:
        logging.info("\n✗ FAIL: Model cannot overfit (loss >= 0.5)")
        logging.info("  Possible issues:")
        logging.info("  - Bug in model architecture")
        logging.info("  - Bug in loss function")
        logging.info("  - Data preprocessing issues")
        logging.info("  - Learning rate too low")
    
    logging.info("="*80)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)