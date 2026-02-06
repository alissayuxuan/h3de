"""
Debug Bad POI Filtering
"""

import pandas as pd
import numpy as np
import ast
from data_utils.spine_dataloader import VertebraePOI
from torchvision import transforms
import data_utils.transforms as tr

# Load data
master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv')

# Find WS-63 vert 6
row = master_df[(master_df['subject'] == 'WS-63') & (master_df['vertebra'] == 6)].iloc[0]

print("="*80)
print("DEBUG: BAD POI FILTERING")
print("="*80)

# Check what's in master_df
print(f"\nMaster DF row:")
print(f"  Subject: {row['subject']}")
print(f"  Vertebra: {row['vertebra']}")
print(f"  File dir: {row['file_dir']}")
if 'bad_poi_list' in master_df.columns:
    print(f"  bad_poi_list (raw): {row['bad_poi_list']}")
    print(f"  bad_poi_list (type): {type(row['bad_poi_list'])}")
    
    # Parse it
    bad_poi_list = ast.literal_eval(row['bad_poi_list'])
    print(f"  bad_poi_list (parsed): {bad_poi_list}")
    print(f"  bad_poi_list (type after parse): {type(bad_poi_list)}")
else:
    print(f"  ⚠️ NO 'bad_poi_list' column in master_df!")

# Now create dataset and check
print(f"\n{'='*80}")
print("Creating dataset...")
print("="*80)

train_transform = transforms.Compose([
    tr.RandomCrop(size=[128, 128, 64], min_rate=0.5),
    tr.LandmarkProposal(size=[128, 128, 64], shrink=4, anchors=[0.5, 0.75, 1., 1.25]),
    tr.ToTensor(),
])

dataset = VertebraePOI(
    master_df=master_df,
    transform=train_transform,
    phase='train',
    input_shape=[215, 215, 144],
    project_gt=False
)

# Find WS-63 vert 6 in dataset
print(f"\nSearching for WS-63 vert 6 in dataset...")
found_idx = None
for idx in range(len(dataset)):
    if dataset.filename[idx] == 'WS-63_vert6_label.npy':
        found_idx = idx
        break

if found_idx is None:
    print("⚠️ WS-63 vert 6 NOT FOUND in dataset!")
else:
    print(f"✓ Found at index {found_idx}")
    
    # Check master_df row for this sample
    dataset_row = dataset.master_df.iloc[found_idx]
    print(f"\nDataset's master_df row:")
    print(f"  Subject: {dataset_row['subject']}")
    print(f"  Vertebra: {dataset_row['vertebra']}")
    if 'bad_poi_list' in dataset.master_df.columns:
        print(f"  bad_poi_list: {dataset_row['bad_poi_list']}")
    else:
        print(f"  ⚠️ NO 'bad_poi_list' column!")
    
    # Get the actual sample
    sample = dataset[found_idx]
    landmarks = sample['landmarks']
    
    print(f"\nLandmarks from __getitem__:")
    print(f"  Shape: {landmarks.shape}")
    
    # Check which POIs are marked as invalid
    invalid_mask = np.all(landmarks == -1000, axis=1)
    invalid_indices = np.where(invalid_mask)[0]
    
    print(f"\nInvalid POIs (marked as -1000):")
    if len(invalid_indices) > 0:
        print(f"  Count: {len(invalid_indices)}/35")
        print(f"  Indices: {invalid_indices}")
        
        # Map back to POI IDs
        poi_indices = dataset.poi_indices
        invalid_poi_ids = [poi_indices[i] for i in invalid_indices]
        print(f"  POI IDs: {invalid_poi_ids}")
    else:
        print(f"  ✗ NONE! All 35 POIs are valid")
    
    # Expected bad POIs
    expected_bad = [81, 82, 83, 85, 87]
    print(f"\nExpected bad POIs (from CSV): {expected_bad}")
    
    # Check if they match
    if len(invalid_indices) > 0:
        if set(invalid_poi_ids) == set(expected_bad):
            print("✓ MATCH: Filtered correctly!")
        else:
            print("✗ MISMATCH!")
            print(f"  Expected: {set(expected_bad)}")
            print(f"  Got:      {set(invalid_poi_ids)}")
            missing = set(expected_bad) - set(invalid_poi_ids)
            extra = set(invalid_poi_ids) - set(expected_bad)
            if missing:
                print(f"  Missing: {missing}")
            if extra:
                print(f"  Extra: {extra}")
    else:
        print("✗ MISMATCH: Expected 5 bad POIs, got 0!")

print(f"\n{'='*80}")
print("CHECKING __getitem__ LOGIC")
print("="*80)

# Manually trace through __getitem__
print("\nManual trace:")
row_idx = dataset.master_df[
    (dataset.master_df['subject'] == 'WS-63') & 
    (dataset.master_df['vertebra'] == 6)
].index[0]

row = dataset.master_df.iloc[row_idx]
print(f"1. Row from master_df: subject={row['subject']}, vertebra={row['vertebra']}")

if 'bad_poi_list' in dataset.master_df.columns:
    bad_poi_list_raw = row['bad_poi_list']
    print(f"2. bad_poi_list (raw): {bad_poi_list_raw} (type: {type(bad_poi_list_raw)})")
    
    bad_poi_list = ast.literal_eval(bad_poi_list_raw)
    print(f"3. bad_poi_list (parsed): {bad_poi_list}")
    
    bad_poi_list = [int(poi) for poi in bad_poi_list]
    print(f"4. bad_poi_list (as ints): {bad_poi_list}")
    
    # Check mapping
    print(f"\n5. Checking poi_idx_to_list_idx mapping:")
    print(f"   dataset.poi_indices: {dataset.poi_indices}")
    print(f"   dataset.poi_idx_to_list_idx: {dataset.poi_idx_to_list_idx}")
    
    print(f"\n6. Marking bad POIs:")
    for bad_poi_id in bad_poi_list:
        if bad_poi_id in dataset.poi_idx_to_list_idx:
            list_idx = dataset.poi_idx_to_list_idx[bad_poi_id]
            print(f"   POI {bad_poi_id} → list index {list_idx} → MARKED")
        else:
            print(f"   POI {bad_poi_id} → NOT in poi_indices! SKIPPED")
else:
    print("2. ⚠️ NO 'bad_poi_list' column in dataset.master_df!")

print("="*80)