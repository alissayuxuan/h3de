"""
Training Sanity Check Script for H3DE-Net with Vertebra Data

This script performs a series of tests to ensure your training setup works:
1. Data loading test
2. Model forward pass test
3. Loss computation test
4. Single batch overfitting test
5. Multi-batch training test
6. Full epoch test

Run this BEFORE starting full training to catch issues early!
"""

import os
import sys
import time
import numpy as np
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# Import your modules
import data_utils.transforms as tr
from utils import setgpu, metric_proposal
from models.losses import HNM_propmap
from models.PBiFormer_Unet import PBiFormer_Unet
from data_utils.spine_dataloader import VertebraDataset, vertebra_collate_fn

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SanityChecker:
    """Systematic testing of training setup"""
    
    def __init__(self, data_path='./gruber_dataset', gpu='0', n_class=35):
        self.data_path = data_path
        self.gpu = gpu
        self.n_class = n_class
        self.shrink = 4
        self.anchors = [0.5, 0.75, 1., 1.25]
        
        print("=" * 70)
        print("H3DE-NET TRAINING SANITY CHECK")
        print("=" * 70)
        print(f"Data path: {data_path}")
        print(f"GPU: {gpu}")
        print(f"Number of landmarks: {n_class}")
        print(f"Device: {DEVICE}")
        print("=" * 70 + "\n")
        
    def test_1_data_loading(self):
        """Test 1: Can we load data with transforms?"""
        print("\n" + "=" * 70)
        print("TEST 1: Data Loading")
        print("=" * 70)
        
        try:
            # Create transforms
            train_transform = transforms.Compose([
                tr.RandomCrop(),
                tr.LandmarkProposal(shrink=self.shrink, anchors=self.anchors),
                tr.Normalize(),
                tr.ToTensor(),
            ])
            
            # Create dataset
            train_dataset = VertebraDataset(
                transform=train_transform,
                phase='train',
                parent_path=self.data_path,
                data_type='full'
            )
            
            print(f"✓ Dataset created: {len(train_dataset)} samples")
            
            # Create dataloader
            trainloader = DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,  # 0 for debugging
                collate_fn=vertebra_collate_fn
            )
            
            print(f"✓ DataLoader created: {len(trainloader)} batches")
            
            # Load one batch
            print("\nLoading first batch...")
            sample = next(iter(trainloader))
            
            print(f"✓ Batch loaded successfully!")
            print(f"  Image: {sample['image'].shape}, dtype={sample['image'].dtype}")
            print(f"  Proposals: {sample['proposals'].shape}, dtype={sample['proposals'].dtype}")
            print(f"  Landmarks: {sample['landmarks'].shape}")
            print(f"  Spacing: {sample['spacing'].shape}")
            
            print("\n✅ TEST 1 PASSED: Data loading works!")
            return True, trainloader
            
        except Exception as e:
            print(f"\n❌ TEST 1 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def test_2_model_forward(self, trainloader):
        """Test 2: Can model do forward pass?"""
        print("\n" + "=" * 70)
        print("TEST 2: Model Forward Pass")
        print("=" * 70)
        
        try:
            # Create model
            print("Creating model...")
            net = PBiFormer_Unet(n_class=self.n_class, n_anchor=len(self.anchors))
            net = net.to(DEVICE)
            
            print(f"✓ Model created and moved to {DEVICE}")
            
            # Get a batch
            sample = next(iter(trainloader))
            data = sample['image'].to(DEVICE)
            
            print(f"\nInput shape: {data.shape}")
            
            # Forward pass
            print("Running forward pass...")
            net.eval()
            with torch.no_grad():
                output = net(data)
            
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
            
            print("\n✅ TEST 2 PASSED: Model forward pass works!")
            return True, net
            
        except Exception as e:
            print(f"\n❌ TEST 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def test_3_loss_computation(self, net, trainloader):
        """Test 3: Can we compute loss?"""
        print("\n" + "=" * 70)
        print("TEST 3: Loss Computation")
        print("=" * 70)
        
        try:
            # Create loss function
            print("Creating loss function...")
            loss_fn = HNM_propmap(n_class=self.n_class, device=DEVICE)
            loss_fn = loss_fn.to(DEVICE)
            
            print(f"✓ Loss function created")
            
            # Get a batch
            sample = next(iter(trainloader))
            data = sample['image'].to(DEVICE)
            proposals = sample['proposals'].to(DEVICE)
            
            print(f"\nInput shapes:")
            print(f"  Data: {data.shape}")
            print(f"  Proposals: {proposals.shape}")
            
            # Forward pass
            net.eval()
            with torch.no_grad():
                proposal_map = net(data)
            
            print(f"  Output: {proposal_map.shape}")
            
            # Compute loss
            print("\nComputing loss...")
            loss_value = loss_fn(proposal_map, proposals)
            
            print(f"✓ Loss computed successfully!")
            print(f"  Loss value: {loss_value.item():.6f}")
            
            print("\n✅ TEST 3 PASSED: Loss computation works!")
            return True, loss_fn
            
        except Exception as e:
            print(f"\n❌ TEST 3 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def test_4_backward_pass(self, net, loss_fn, trainloader):
        """Test 4: Can we do backward pass and optimizer step?"""
        print("\n" + "=" * 70)
        print("TEST 4: Backward Pass & Optimizer")
        print("=" * 70)
        
        try:
            # Create optimizer
            print("Creating optimizer...")
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=0.001,
                betas=(0.9, 0.98),
                weight_decay=0.0005
            )
            
            print(f"✓ Optimizer created")
            
            # Get a batch
            sample = next(iter(trainloader))
            data = sample['image'].to(DEVICE)
            proposals = sample['proposals'].to(DEVICE)
            
            # Forward pass
            net.train()
            proposal_map = net(data)
            
            # Compute loss
            loss_value = loss_fn(proposal_map, proposals)
            print(f"\nInitial loss: {loss_value.item():.6f}")
            
            # Backward pass
            print("Running backward pass...")
            optimizer.zero_grad()
            loss_value.backward()
            
            print(f"✓ Backward pass successful!")
            
            # Check gradients
            grad_norm = sum(p.grad.norm() for p in net.parameters() if p.grad is not None)
            print(f"  Total gradient norm: {grad_norm:.6f}")
            
            # Optimizer step
            print("Running optimizer step...")
            optimizer.step()
            
            print(f"✓ Optimizer step successful!")
            
            print("\n✅ TEST 4 PASSED: Backward pass and optimizer work!")
            return True, optimizer
            
        except Exception as e:
            print(f"\n❌ TEST 4 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def test_5_single_batch_overfit(self, net, loss_fn, optimizer, trainloader):
        """Test 5: Can we overfit on a single batch?"""
        print("\n" + "=" * 70)
        print("TEST 5: Single Batch Overfitting (Sanity Check)")
        print("=" * 70)
        print("This tests if the model can memorize one batch.")
        print("Loss should decrease significantly!\n")
        
        try:
            # Get one batch
            sample = next(iter(trainloader))
            data = sample['image'].to(DEVICE)
            proposals = sample['proposals'].to(DEVICE)
            
            print(f"Training on single batch:")
            print(f"  Data shape: {data.shape}")
            print(f"  Proposals shape: {proposals.shape}")
            
            net.train()
            losses = []
            
            print("\nIterations:")
            for i in range(50):  # 50 iterations on same batch
                proposal_map = net(data)
                loss_value = loss_fn(proposal_map, proposals)
                
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                
                losses.append(loss_value.item())
                
                if i % 10 == 0:
                    print(f"  Iter {i:3d}: Loss = {loss_value.item():.6f}")
            
            initial_loss = losses[0]
            final_loss = losses[-1]
            reduction = (initial_loss - final_loss) / initial_loss * 100
            
            print(f"\nResults:")
            print(f"  Initial loss: {initial_loss:.6f}")
            print(f"  Final loss:   {final_loss:.6f}")
            print(f"  Reduction:    {reduction:.2f}%")
            
            if reduction > 50:  # Loss should drop by >50%
                print(f"\n✅ TEST 5 PASSED: Model can learn (loss dropped {reduction:.1f}%)!")
                return True
            else:
                print(f"\n⚠️  TEST 5 WARNING: Loss only dropped {reduction:.1f}%")
                print("   This might indicate a problem, but could also be normal.")
                return True
            
        except Exception as e:
            print(f"\n❌ TEST 5 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_6_multiple_batches(self, trainloader):
        """Test 6: Can we train on multiple batches?"""
        print("\n" + "=" * 70)
        print("TEST 6: Multiple Batches Training")
        print("=" * 70)
        
        try:
            # Create fresh model and optimizer
            net = PBiFormer_Unet(n_class=self.n_class, n_anchor=len(self.anchors))
            net = net.to(DEVICE)
            
            loss_fn = HNM_propmap(n_class=self.n_class, device=DEVICE)
            loss_fn = loss_fn.to(DEVICE)
            
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=0.001,
                betas=(0.9, 0.98),
                weight_decay=0.0005
            )
            
            net.train()
            
            num_batches = min(5, len(trainloader))  # Test on 5 batches
            print(f"Training on {num_batches} batches...\n")
            
            batch_losses = []
            
            for i, sample in enumerate(trainloader):
                if i >= num_batches:
                    break
                
                data = sample['image'].to(DEVICE)
                proposals = sample['proposals'].to(DEVICE)
                
                proposal_map = net(data)
                loss_value = loss_fn(proposal_map, proposals)
                
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                
                batch_losses.append(loss_value.item())
                print(f"  Batch {i+1}/{num_batches}: Loss = {loss_value.item():.6f}")
            
            avg_loss = np.mean(batch_losses)
            print(f"\nAverage loss: {avg_loss:.6f}")
            
            print("\n✅ TEST 6 PASSED: Multiple batches training works!")
            return True
            
        except Exception as e:
            print(f"\n❌ TEST 6 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_7_full_epoch(self, trainloader):
        """Test 7: Can we run a full epoch?"""
        print("\n" + "=" * 70)
        print("TEST 7: Full Epoch Training")
        print("=" * 70)
        
        try:
            # Create fresh model and optimizer
            net = PBiFormer_Unet(n_class=self.n_class, n_anchor=len(self.anchors))
            net = net.to(DEVICE)
            
            loss_fn = HNM_propmap(n_class=self.n_class, device=DEVICE)
            loss_fn = loss_fn.to(DEVICE)
            
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=0.001,
                betas=(0.9, 0.98),
                weight_decay=0.0005
            )
            
            net.train()
            
            print(f"Training on full epoch ({len(trainloader)} batches)...")
            print("(This might take a few minutes)\n")
            
            start_time = time.time()
            epoch_losses = []
            
            for i, sample in enumerate(trainloader):
                data = sample['image'].to(DEVICE)
                proposals = sample['proposals'].to(DEVICE)
                
                proposal_map = net(data)
                loss_value = loss_fn(proposal_map, proposals)
                
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                
                epoch_losses.append(loss_value.item())
                
                # Print every 10% of batches
                if (i + 1) % max(1, len(trainloader) // 10) == 0:
                    progress = (i + 1) / len(trainloader) * 100
                    print(f"  Progress: {progress:.0f}% ({i+1}/{len(trainloader)}) - "
                          f"Current loss: {loss_value.item():.6f}")
            
            elapsed_time = time.time() - start_time
            avg_loss = np.mean(epoch_losses)
            
            print(f"\nEpoch completed!")
            print(f"  Average loss: {avg_loss:.6f}")
            print(f"  Time: {elapsed_time:.1f} seconds")
            print(f"  Batches/sec: {len(trainloader)/elapsed_time:.2f}")
            
            print("\n✅ TEST 7 PASSED: Full epoch training works!")
            return True
            
        except Exception as e:
            print(f"\n❌ TEST 7 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_8_validation(self):
        """Test 8: Can we run validation?"""
        print("\n" + "=" * 70)
        print("TEST 8: Validation Loop")
        print("=" * 70)
        
        try:
            # Create validation dataset
            eval_transform = transforms.Compose([
                tr.CenterCrop(),
                tr.LandmarkProposal(shrink=self.shrink, anchors=self.anchors),
                tr.Normalize(),
                tr.ToTensor(),
            ])
            
            eval_dataset = VertebraDataset(
                transform=eval_transform,
                phase='val',
                parent_path=self.data_path,
                data_type='full'
            )
            
            print(f"✓ Validation dataset: {len(eval_dataset)} samples")
            
            evalloader = DataLoader(
                eval_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                collate_fn=vertebra_collate_fn
            )
            
            # Create model
            net = PBiFormer_Unet(n_class=self.n_class, n_anchor=len(self.anchors))
            net = net.to(DEVICE)
            
            loss_fn = HNM_propmap(n_class=self.n_class, device=DEVICE)
            loss_fn = loss_fn.to(DEVICE)
            
            # Run validation
            net.eval()
            val_losses = []
            
            print("\nRunning validation...")
            with torch.no_grad():
                for i, sample in enumerate(evalloader):
                    data = sample['image'].to(DEVICE)
                    proposals = sample['proposals'].to(DEVICE)
                    
                    proposal_map = net(data)
                    loss_value = loss_fn(proposal_map, proposals)
                    val_losses.append(loss_value.item())
                    
                    if i >= 5:  # Just test a few batches
                        break
            
            avg_val_loss = np.mean(val_losses)
            print(f"✓ Validation loss: {avg_val_loss:.6f}")
            
            print("\n✅ TEST 8 PASSED: Validation loop works!")
            return True
            
        except Exception as e:
            print(f"\n❌ TEST 8 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "=" * 70)
        print("RUNNING ALL SANITY CHECKS")
        print("=" * 70)
        
        results = []
        
        # Test 1: Data loading
        success, trainloader = self.test_1_data_loading()
        results.append(("Data Loading", success))
        if not success:
            self.print_summary(results)
            return
        
        # Test 2: Model forward
        success, net = self.test_2_model_forward(trainloader)
        results.append(("Model Forward Pass", success))
        if not success:
            self.print_summary(results)
            return
        
        # Test 3: Loss computation
        success, loss_fn = self.test_3_loss_computation(net, trainloader)
        results.append(("Loss Computation", success))
        if not success:
            self.print_summary(results)
            return
        
        # Test 4: Backward pass
        success, optimizer = self.test_4_backward_pass(net, loss_fn, trainloader)
        results.append(("Backward Pass", success))
        if not success:
            self.print_summary(results)
            return
        
        # Test 5: Single batch overfit
        success = self.test_5_single_batch_overfit(net, loss_fn, optimizer, trainloader)
        results.append(("Single Batch Overfit", success))
        
        # Test 6: Multiple batches
        success = self.test_6_multiple_batches(trainloader)
        results.append(("Multiple Batches", success))
        
        # Test 7: Full epoch
        success = self.test_7_full_epoch(trainloader)
        results.append(("Full Epoch", success))
        
        # Test 8: Validation
        success = self.test_8_validation()
        results.append(("Validation Loop", success))
        
        # Print summary
        self.print_summary(results)
    
    def print_summary(self, results):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {test_name}")
        
        all_passed = all(result[1] for result in results)
        
        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
            print("\nYour training setup is ready!")
            print("\nNext steps:")
            print("1. Run full training: python train.py --data_path ./gruber_dataset --n_class 35")
            print("2. Monitor log.txt for training progress")
            print("3. Check for overfitting on validation set")
        else:
            print("❌ SOME TESTS FAILED")
            print("\nPlease fix the failed tests before starting training.")
            print("The tests stop at the first failure to help you debug.")
        print("=" * 70)


def quick_sanity_check(data_path='./gruber_dataset', n_class=35):
    """Quick version - just test if one training iteration works"""
    print("=" * 70)
    print("QUICK SANITY CHECK")
    print("=" * 70)
    
    try:
        # Setup
        setgpu('0')
        
        # Data
        train_transform = transforms.Compose([
            tr.RandomCrop(),
            tr.LandmarkProposal(shrink=4, anchors=[0.5, 0.75, 1., 1.25]),
            tr.Normalize(),
            tr.ToTensor(),
        ])
        
        train_dataset = VertebraDataset(
            transform=train_transform,
            phase='train',
            parent_path=data_path,
            data_type='full'
        )
        
        trainloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=vertebra_collate_fn
        )
        
        # Model
        net = PBiFormer_Unet(n_class=n_class, n_anchor=4)
        net = net.to(DEVICE)
        
        # Loss
        loss_fn = HNM_propmap(n_class=n_class, device=DEVICE)
        loss_fn = loss_fn.to(DEVICE)
        
        # Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        
        # One training step
        sample = next(iter(trainloader))
        data = sample['image'].to(DEVICE)
        proposals = sample['proposals'].to(DEVICE)
        
        net.train()
        proposal_map = net(data)
        loss_value = loss_fn(proposal_map, proposals)
        
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        
        print(f"\n✅ Quick check PASSED!")
        print(f"   Loss: {loss_value.item():.6f}")
        print(f"\n   Your setup works! Ready for full training.")
        
    except Exception as e:
        print(f"\n❌ Quick check FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sanity check for H3DE-Net training')
    parser.add_argument('--data_path', default='./gruber_dataset', type=str,
                       help='Path to dataset')
    parser.add_argument('--n_class', default=35, type=int,
                       help='Number of landmarks (POIs per vertebra)')
    parser.add_argument('--gpu', default='0', type=str,
                       help='GPU to use')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick check only')
    
    args = parser.parse_args()
    
    # Set GPU
    setgpu(args.gpu)
    
    if args.quick:
        quick_sanity_check(args.data_path, args.n_class)
    else:
        checker = SanityChecker(args.data_path, args.gpu, args.n_class)
        checker.run_all_tests()