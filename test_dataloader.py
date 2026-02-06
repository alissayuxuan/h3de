
import pandas as pd
from data_utils.spine_dataloader import VertebraePOI
from torch.utils.data import DataLoader
import data_utils.transforms as tr
from torchvision import transforms
from data_utils.dataloader import Molar3D



# Lade master_df
master_df = pd.read_csv('datasets/gruber-cutouts-fixed_size/master_df-excel_outliers_proj_6_exclude-split.csv')
#master_df = pd.read_csv('datasets/gruber_dataset_cutouts/cutouts/master_df.csv')

#train_df = master_df[master_df['split'] == 'train']

#print(f"Train samples: {len(train_df)}")
#print(f"Subjects: {train_df['subject'].nunique()}")

train_transform = transforms.Compose([
    tr.RandomCrop(size=[128, 128, 64], min_rate=0.5),
    tr.LandmarkProposal(size=[128, 128, 64], shrink=4, anchors=[0.5, 0.75, 1., 1.25]),
    #tr.Normalize(),
    tr.ToTensor(),
])

dataset_with_transforms = VertebraePOI(
    master_df=master_df,#train_df,
    transform=train_transform,
    phase='train',
    input_data_type='ct',
    poi_indices=[
            81, 82, 83, 84, 85, 86, 87, 88, 89,
            101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 114, 115, 116,
            117, 118, 119, 120, 121, 122, 123, 124,
            125, 127
        ],
    input_shape=(215, 215, 144)#128),
)

# Erstelle Dataset


import torch
from models.PBiFormer_Unet import PBiFormer_Unet
from models.losses import HNM_propmap

# Mini-Dataset: NUR 1 Sample!
#train_df_mini = train_df.iloc[:5].copy()

"""dataset_mini = VertebraePOI(
    master_df=train_df_mini,
    transform=train_transform,
    phase='train',
    input_data_type='ct',
    poi_indices=[
            81, 82, 83, 84, 85, 86, 87, 88, 89,
            101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 114, 115, 116,
            117, 118, 119, 120, 121, 122, 123, 124,
            125, 127
        ],
    input_shape=(215, 215, 144),
)"""

"""
dataset_molar = Molar3D(transform=train_transform,
                      phase='test',
                      parent_path='datasets/mmld_dataset',
                      data_type ="full")"""

#loader_mini = DataLoader(dataset_molar, batch_size=1, shuffle=False)
loader_mini = DataLoader(dataset_mini, batch_size=1, shuffle=False)

#sample = dataset_molar[0]#dataset_mini[0]
sample = dataset_mini[0]
print(f"\n=== DEBUG ===")
print(f"Sample image shape: {sample['image'].shape}")
print(f"Sample image dtype: {sample['image'].dtype}")
print(f"Sample proposals shape: {sample['proposals'].shape}")




# Model & Loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = PBiFormer_Unet(n_class=35, n_anchor=4).to(device)
loss_fn = HNM_propmap(n_class=35, device=device).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Overfitting Test: 100 Iterationen
net.train()
losses = []

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
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Plot
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Overfitting Test - Loss sollte gegen 0 gehen!')
plt.savefig('overfit_test.png')
plt.show()

print(f"\nFinal loss: {losses[-1]:.6f}")
print(f"Initial loss: {losses[0]:.6f}")
print(f"Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
