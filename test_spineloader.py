import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from data_utils.spine_dataloader import VertebraePOI
import data_utils.transforms as tr
from torchvision import transforms


# ============================================================================
# TEST 1: Basic Loading & Shapes
# ============================================================================
def test_basic_loading():
    """Test ob der Dataloader grundsätzlich lädt"""
    print("\n" + "="*80)
    print("TEST 1: Basic Loading & Shapes")
    print("="*80)
    
    master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv')
    
    # Ohne Transforms
    dataset = VertebraePOI(
        master_df=master_df,
        transform=None,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        project_gt=False
    )
    
    print(f"✓ Dataset length: {len(dataset)}")
    
    # Load one sample
    sample = dataset[0]
    
    print(f"✓ Image shape: {sample['image'].shape}")
    print(f"✓ Image dtype: {sample['image'].dtype}")
    print(f"✓ Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"✓ Landmarks shape: {sample['landmarks'].shape}")
    print(f"✓ Landmarks dtype: {sample['landmarks'].dtype}")
    print(f"✓ Spacing: {sample['spacing']}")
    print(f"✓ Filename: {sample['filename']}")
    
    # Check expected shapes
    assert sample['image'].shape == (215, 215, 144), "Wrong image shape!"
    assert sample['landmarks'].shape == (35, 3), "Wrong landmarks shape!"
    assert sample['image'].dtype == np.float64, "Wrong image dtype!"
    
    print("✓ All basic checks passed!")
    return dataset


# ============================================================================
# TEST 2: Split Consistency
# ============================================================================
def test_split_consistency():
    """Test ob train/val/test splits korrekt sind"""
    print("\n" + "="*80)
    print("TEST 2: Split Consistency")
    print("="*80)
    
    master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv')
    
    splits = {}
    for phase in ['train', 'val', 'test']:
        dataset = VertebraePOI(
            master_df=master_df,
            transform=None,
            phase=phase,
            input_data_type='ct',
            input_shape=(215, 215, 144),
            project_gt=False
        )
        splits[phase] = {
            'size': len(dataset),
            'subjects': set(dataset.master_df['subject'].unique())
        }
    
    print(f"✓ Train: {splits['train']['size']} samples, {len(splits['train']['subjects'])} subjects")
    print(f"✓ Val:   {splits['val']['size']} samples, {len(splits['val']['subjects'])} subjects")
    print(f"✓ Test:  {splits['test']['size']} samples, {len(splits['test']['subjects'])} subjects")
    
    # Check no subject overlap
    train_val_overlap = splits['train']['subjects'] & splits['val']['subjects']
    train_test_overlap = splits['train']['subjects'] & splits['test']['subjects']
    val_test_overlap = splits['val']['subjects'] & splits['test']['subjects']
    
    assert len(train_val_overlap) == 0, f"Train/Val overlap: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Train/Test overlap: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Val/Test overlap: {val_test_overlap}"
    
    print("✓ No subject leakage between splits!")
    return splits


# ============================================================================
# TEST 3: Invalid POI Marking
# ============================================================================
def test_invalid_poi_marking():
    """Test ob invalid POIs korrekt mit -1000 markiert werden"""
    print("\n" + "="*80)
    print("TEST 3: Invalid POI Marking")
    print("="*80)
    
    master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv')
    
    dataset = VertebraePOI(
        master_df=master_df,
        transform=None,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        project_gt=False
    )
    
    # Check multiple samples
    n_samples = min(10, len(dataset))
    invalid_counts = []
    
    for i in range(n_samples):
        sample = dataset[i]
        landmarks = sample['landmarks']
        
        # Count invalid POIs (-1000, -1000, -1000)
        invalid_mask = np.all(landmarks == -1000, axis=1)
        n_invalid = invalid_mask.sum()
        invalid_counts.append(n_invalid)
        
        if n_invalid > 0:
            print(f"  Sample {i}: {n_invalid}/35 POIs marked as invalid")
    
    print(f"\n✓ Checked {n_samples} samples")
    print(f"✓ Average invalid POIs: {np.mean(invalid_counts):.1f}")
    print(f"✓ Max invalid POIs: {np.max(invalid_counts)}")
    
    return invalid_counts


# ============================================================================
# TEST 4: Transform Pipeline
# ============================================================================
def test_transforms():
    """Test ob Transforms korrekt angewendet werden"""
    print("\n" + "="*80)
    print("TEST 4: Transform Pipeline")
    print("="*80)
    
    master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df_split.csv')
    
    train_transform = transforms.Compose([
        tr.RandomCrop(size=[128, 128, 64], min_rate=0.5),
        tr.LandmarkProposal(size=[128, 128, 64], shrink=4, anchors=[0.5, 0.75, 1., 1.25]),
        tr.ToTensor(),  # Konvertiert zu numpy array für H3DE-Net!
    ])
    
    dataset = VertebraePOI(
        master_df=master_df,
        transform=train_transform,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        project_gt=False
    )
    
    sample = dataset[0]
    
    print(f"✓ After transforms:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Image dtype: {sample['image'].dtype}")
    print(f"  Proposals shape: {sample['proposals'].shape}")
    
    # ===== KORRIGIERT: Check numpy array type =====
    # H3DE-Net's ToTensor() konvertiert zu numpy, nicht torch.Tensor!
    assert isinstance(sample['image'], np.ndarray), "Image should be numpy array!"
    assert isinstance(sample['proposals'], np.ndarray), "Proposals should be numpy array!"
    
    # Nach DataLoader werden sie zu Tensoren
    print("✓ Note: After DataLoader, these become torch.Tensors")
    
    print("✓ Transform pipeline works!")
    return sample

def test_transforms_old():
    """Test ob Transforms korrekt angewendet werden"""
    print("\n" + "="*80)
    print("TEST 4: Transform Pipeline")
    print("="*80)
    
    master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv')
    
    train_transform = transforms.Compose([
        tr.RandomCrop(size=[128, 128, 64], min_rate=0.5),
        tr.LandmarkProposal(size=[128, 128, 64], shrink=4, anchors=[0.5, 0.75, 1., 1.25]),
        # NO Normalize() - wir normalisieren bereits im dataloader!
        tr.ToTensor(),
    ])
    
    dataset = VertebraePOI(
        master_df=master_df,
        transform=train_transform,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        project_gt=False
    )
    
    sample = dataset[0]
    
    print(f"✓ After transforms:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Image dtype: {sample['image'].dtype}")
    print(f"  Proposals shape: {sample['proposals'].shape}")
    
    # Check tensor type
    assert isinstance(sample['image'], torch.Tensor), "Image not converted to tensor!"
    assert isinstance(sample['proposals'], torch.Tensor), "Proposals not converted to tensor!"
    
    print("✓ Transform pipeline works!")
    return sample


# ============================================================================
# TEST 5: DataLoader Batching
# ============================================================================
def test_dataloader_batching():
    """Test ob DataLoader korrekt batched"""
    print("\n" + "="*80)
    print("TEST 5: DataLoader Batching")
    print("="*80)
    
    master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv')
    
    train_transform = transforms.Compose([
        tr.RandomCrop(size=[128, 128, 64], min_rate=0.5),
        tr.LandmarkProposal(size=[128, 128, 64], shrink=4, anchors=[0.5, 0.75, 1., 1.25]),
        tr.ToTensor(),
    ])
    
    dataset = VertebraePOI(
        master_df=master_df,
        transform=train_transform,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        project_gt=False
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    batch = next(iter(loader))
    
    print(f"✓ Batch image shape: {batch['image'].shape}")
    print(f"✓ Batch proposals shape: {batch['proposals'].shape}")
    print(f"✓ Batch spacing shape: {batch['spacing'].shape}")
    print(f"✓ Batch filenames: {len(batch['filename'])} items")
    
    assert batch['image'].shape[0] == 4, "Wrong batch size!"
    
    print("✓ DataLoader batching works!")
    return loader


# ============================================================================
# TEST 6: Surface Projection (NEW!)
# ============================================================================
def test_surface_projection():
    """Test ob Surface Projection korrekt funktioniert"""
    print("\n" + "="*80)
    print("TEST 6: Surface Projection")
    print("="*80)
    
    master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv')
    
    # Without projection
    dataset_no_proj = VertebraePOI(
        master_df=master_df,
        transform=None,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        project_gt=False
    )
    
    # With projection
    dataset_with_proj = VertebraePOI(
        master_df=master_df,
        transform=None,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        project_gt=True,
    )
    
    # Compare same sample
    sample_no_proj = dataset_no_proj[0]
    sample_with_proj = dataset_with_proj[0]
    
    landmarks_orig = sample_no_proj['landmarks']
    landmarks_proj = sample_with_proj['landmarks']
    
    # Calculate projection distances
    valid_mask = ~np.all(landmarks_orig == -1000, axis=1)
    distances = np.linalg.norm(landmarks_orig - landmarks_proj, axis=1)
    valid_distances = distances[valid_mask]
    
    print(f"✓ Original landmarks shape: {landmarks_orig.shape}")
    print(f"✓ Projected landmarks shape: {landmarks_proj.shape}")
    print(f"✓ Valid POIs: {valid_mask.sum()}/35")
    print(f"✓ Mean projection distance: {valid_distances.mean():.3f} voxels")
    print(f"✓ Max projection distance: {valid_distances.max():.3f} voxels")
    print(f"✓ Min projection distance: {valid_distances.min():.3f} voxels")
    
    # Check that projection actually changed coordinates
    assert not np.allclose(landmarks_orig[valid_mask], landmarks_proj[valid_mask]), \
        "Projection didn't change coordinates!"
    
    print("✓ Surface projection works!")
    return valid_distances


# ============================================================================
# TEST 7: Overfitting Test (Mini-Batch)
# ============================================================================
def test_overfitting():
    """Test ob Modell auf 1 Sample overfitted"""
    print("\n" + "="*80)
    print("TEST 7: Overfitting Test")
    print("="*80)
    
    master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df_split.csv')
    
    train_transform = transforms.Compose([
        tr.RandomCrop(size=[128, 128, 64], min_rate=0.5),
        tr.LandmarkProposal(size=[128, 128, 64], shrink=4, anchors=[0.5, 0.75, 1., 1.25]),
        tr.ToTensor(),
    ])
    
    # ===== FIX: Lade erst normales Dataset, dann nimm 1 Sample =====
    # Option 1: Verwende komplettes train set, aber batch_size=1 und nur 1 Sample
    dataset_full = VertebraePOI(
        master_df=master_df,
        transform=train_transform,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        project_gt=False
    )
    
    # Erstelle Subset mit nur 1 Sample
    from torch.utils.data import Subset
    dataset_mini = Subset(dataset_full, [0])  # Nur Index 0
    
    print(f"✓ Mini dataset size: {len(dataset_mini)}")
    
    loader_mini = DataLoader(dataset_mini, batch_size=1, shuffle=False)
    
    # Model & Loss
    from models.PBiFormer_Unet import PBiFormer_Unet
    from models.losses import HNM_propmap
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PBiFormer_Unet(n_class=35, n_anchor=4).to(device)
    loss_fn = HNM_propmap(n_class=35, device=device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Train for 100 iterations
    net.train()
    losses = []
    
    print("Training for 100 iterations on 1 sample...")
    for epoch in range(100):
        for sample in loader_mini:
            data = sample['image'].to(device)
            proposals = sample['proposals'].to(device)
            
            # Forward
            proposal_map = net(data)
            
            # Backward
            optimizer.zero_grad()
            loss = loss_fn(proposal_map, proposals)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Overfitting Test - Loss sollte gegen 0 gehen!')
    plt.grid(True)
    plt.savefig('overfit_test_new_dataloader.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: overfit_test_new_dataloader.png")
    
    print(f"\n✓ Final loss: {losses[-1]:.6f}")
    print(f"✓ Initial loss: {losses[0]:.6f}")
    print(f"✓ Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    # Check if loss decreased significantly
    assert losses[-1] < losses[0] * 0.1, "Loss didn't decrease enough - model can't overfit!"
    
    print("✓ Overfitting test passed!")
    return losses

# ============================================================================
# TEST 8: Neighbor Mode
# ============================================================================
def test_neighbor_mode():
    """Test ob show_neighbors korrekt funktioniert"""
    print("\n" + "="*80)
    print("TEST 8: Neighbor Mode")
    print("="*80)
    
    master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv')
    
    # Without neighbors
    dataset_no_neighbors = VertebraePOI(
        master_df=master_df,
        transform=None,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        show_neighbors=False
    )
    
    # With neighbors
    dataset_with_neighbors = VertebraePOI(
        master_df=master_df,
        transform=None,
        phase='train',
        input_data_type='ct',
        input_shape=(215, 215, 144),
        show_neighbors=True,
        neighbor_drop_prob=0.0  # No augmentation for testing
    )
    
    # Compare masks
    sample_no = dataset_no_neighbors[10]  # Pick middle vertebra
    sample_yes = dataset_with_neighbors[10]
    
    img_no = sample_no['image']
    img_yes = sample_yes['image']
    
    # Count non-zero voxels (should be more with neighbors)
    n_voxels_no = (img_no > 0).sum()
    n_voxels_yes = (img_yes > 0).sum()
    
    print(f"✓ Without neighbors: {n_voxels_no} non-zero voxels")
    print(f"✓ With neighbors: {n_voxels_yes} non-zero voxels")
    print(f"✓ Ratio: {n_voxels_yes / n_voxels_no:.2f}x")
    
    assert n_voxels_yes > n_voxels_no, "Neighbor mode didn't increase visible voxels!"
    
    print("✓ Neighbor mode works!")


# ============================================================================
# RUN ALL TESTS
# ============================================================================
def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RUNNING ALL DATALOADER TESTS")
    print("="*80)
    
    results = {}
    
    try:
        results['basic'] = test_basic_loading()
        print("✓ Test 1 passed")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
    
    try:
        results['splits'] = test_split_consistency()
        print("✓ Test 2 passed")
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
    
    try:
        results['invalid_pois'] = test_invalid_poi_marking()
        print("✓ Test 3 passed")
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
    
    try:
        results['transforms'] = test_transforms()
        print("✓ Test 4 passed")
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
    
    try:
        results['batching'] = test_dataloader_batching()
        print("✓ Test 5 passed")
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}")
    
    try:
        results['projection'] = test_surface_projection()
        print("✓ Test 6 passed")
    except Exception as e:
        print(f"✗ Test 6 FAILED: {e}")
    
    try:
        results['neighbors'] = test_neighbor_mode()
        print("✓ Test 8 passed")
    except Exception as e:
        print(f"✗ Test 8 FAILED: {e}")
    
    # Overfitting test last (takes time)
    try:
        results['overfitting'] = test_overfitting()
        print("✓ Test 7 passed")
    except Exception as e:
        print(f"✗ Test 7 FAILED: {e}")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()