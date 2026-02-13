import ast
import os
import numpy as np
from torch.utils.data import Dataset
import json
import torch
from TPTBox import NII, POI
from pathlib import Path
import pandas as pd
from skimage.morphology import binary_erosion

from misc import surface_project_coords



def pad_array_to_shape(arr, target_shape):
    """Pads the input array arr to the target_shape. The original array is centered
    within the new shape.

    Parameters:
    - arr: numpy array of shape (H, W, D)
    - target_shape: tuple of the target shape (H', W', D')

    Returns:
    - Padded numpy array of shape (H', W', D')
    """
    # Calculate the padding needed for each dimension
    pad_h = (target_shape[0] - arr.shape[0]) // 2
    pad_w = (target_shape[1] - arr.shape[1]) // 2
    pad_d = (target_shape[2] - arr.shape[2]) // 2

    # Handle odd differences by adding an extra padding at the end if necessary
    pad_h2 = pad_h + (target_shape[0] - arr.shape[0]) % 2
    pad_w2 = pad_w + (target_shape[1] - arr.shape[1]) % 2
    pad_d2 = pad_d + (target_shape[2] - arr.shape[2]) % 2

    # Apply padding
    padded_arr = np.pad(
        arr, ((pad_h, pad_h2), (pad_w, pad_w2), (pad_d, pad_d2)), mode="constant"
    )

    offset = (pad_h, pad_w, pad_d)

    return padded_arr, offset
    

def get_gt_pois(poi, vertebra, poi_indices):
    """Converts the POI coordinates to a tensor.

    Args:
        poi (POI): The POI coordinates.
        vertebra (int): The vertebra number.

    Returns:
        torch.Tensor: The POI coordinates as a tensor.
    """
    coords = [
        (
            np.array((-1, -1, -1))
            if not (vertebra, p_idx) in poi.keys()
            else np.array(poi.centroids[vertebra, p_idx])
        )
        for p_idx in poi_indices
    ]

    # Stack the coordinates
    coords = np.stack(coords)

    # Change type of coords to float
    coords = coords.astype(np.float32)  # Shape: (n_pois, 3)

    # Mark the missing pois
    missing_poi_list_idx = np.all(coords == -1, axis=1)  # Shape: (n_pois,)

    # Get the indices of missing pois
    missing_pois = np.array(
        [poi_idx for i, poi_idx in enumerate(poi_indices) if missing_poi_list_idx[i]]
    )

    return torch.from_numpy(coords), torch.from_numpy(missing_pois)


def compute_surface(msk: torch.tensor, iterations=1) -> torch.tensor:
    """Computes the surface of the vertebra.

    Args:
        msk (numpy.ndarray): The segmentation mask.
        vertebra (int): The vertebra number.

    Returns:
        torch.Tensor: The surface of the vertebra.
    """
    surface = msk.numpy()

    eroded = surface.copy()
    for _ in range(iterations):
        eroded = binary_erosion(eroded)

    surface[eroded] = 0

    return torch.from_numpy(surface)


class VertebraePOI(Dataset):
    def __init__(
        self,
        master_df,
        transform=None,
        phase='train',
        data_type="full",
        input_shape=(215, 215, 128),#(128, 128, 96),
        zoom=(1, 1, 1),
        poi_indices=None,
        input_data_type="ct",
        poi_file_ending="poi.json",
        show_neighbors=False,
        neighbor_drop_prob=0.0,
        include_vert_list=None,
        project_gt=False,
        surface_erosion_iterations=1
    ):
        """
        Dataset für Wirbel-POI Detektion - kompatibel mit H3DE-Net Baseline
        
        Args:
            master_df: DataFrame mit subject, vertebra, file_dir columns
            transform: Augmentation transforms (aus data_utils)
            phase: 'train', 'val', or 'test'
            parent_path: Wird ignoriert, nutzt master_df stattdessen
            data_type: "full" oder "mini" (für Kompatibilität mit Original)
            input_shape: Target shape für Cropping
            zoom: Voxel spacing (x, y, z)
            poi_indices: Liste der POI IDs zum Predicten
            input_data_type: "ct", "vertseg", "subreg", "surface_msk"
            poi_file_ending: Name der POI-Datei
        """


        # Filter master_df if needed
        if "use_sample" in master_df.columns:
            master_df = master_df[master_df["use_sample"]]
        
        # Filter by phase
        master_df = master_df[master_df['split'] == phase].reset_index(drop=True)
        
        # POI indices
        self.poi_indices = poi_indices if poi_indices is not None else [
            81, 82, 83, 84, 85, 86, 87, 88, 89,
            101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 114, 115, 116,
            117, 118, 119, 120, 121, 122, 123, 124,
            125, 127
        ]

        self.include_vert_list = include_vert_list if include_vert_list is not None else [
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ]

        # Filter by include_vert_list
        master_df = master_df[master_df['vertebra'].isin(self.include_vert_list)].reset_index(drop=True)
        
        # Initialize lists
        self.data_files = []
        self.label_files = []
        self.spacing = []
        self.filename = []
        valid_rows = []  
        skipped_samples = []
                
        # ===== Filter UND baue Listen gleichzeitig auf =====
        for idx, row in master_df.iterrows():
            subject = row["subject"]
            vertebra = row["vertebra"]
            file_dir = row["file_dir"]
            
            ct_path = os.path.join(file_dir, "ct.nii.gz")
            poi_path = os.path.join(file_dir, poi_file_ending)
            vertseg_path = os.path.join(file_dir, "vertseg.nii.gz")
            
            # Check files exist
            if not (os.path.exists(ct_path) and os.path.exists(poi_path) and 
                    os.path.exists(vertseg_path)):
                skipped_samples.append(f"{subject}_vert{vertebra} (missing files)")
                continue
            
            # Check valid POIs
            try:
                poi = POI.load(poi_path)
                poi.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_(
                    zoom, verbose=False
                )
                
                n_valid_pois = sum(
                    1 for p_idx in self.poi_indices 
                    if (vertebra, p_idx) in poi.keys()
                )
                
                if n_valid_pois == 0:
                    skipped_samples.append(f"{subject}_vert{vertebra} (0 valid POIs)")
                    continue
                    
            except Exception as e:
                skipped_samples.append(f"{subject}_vert{vertebra} (error: {e})")
                continue
            
            # Sample is valid - add to lists
            ct_nii = NII.load(ct_path, seg=False)
            spacing_mm = ct_nii.zoom
            
            self.data_files.append(file_dir)
            self.label_files.append(poi_path)
            self.spacing.append(spacing_mm)
            self.filename.append(f"{subject}_vert{vertebra}_label.npy")
            valid_rows.append(row)  # ← Sammle valide row
        
        # Erstelle gefilterten DataFrame
        self.master_df = pd.DataFrame(valid_rows).reset_index(drop=True)
        
        # Report
        if len(skipped_samples) > 0:
            print(f"Skipped {len(skipped_samples)} samples:")
            for s in skipped_samples[:10]:
                print(f"  - {s}")
            if len(skipped_samples) > 10:
                print(f"  ... and {len(skipped_samples) - 10} more")
        
        print(f"Final data length: {len(self.data_files)} for {phase}")
        
        # Validation
        if len(self.master_df) == 0:
            raise ValueError(f"Hallo, No samples with valid POIs found for phase '{phase}'!")
        
        # Store other params
        self.transform = transform
        self.input_shape = input_shape
        self.zoom = zoom
        self.input_data_type = input_data_type
        self.poi_file_ending = poi_file_ending
        self.show_neighbors = show_neighbors
        self.neighbor_drop_prob = neighbor_drop_prob
        self.project_gt = project_gt
        self.surface_erosion_iterations = surface_erosion_iterations 

        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(self.poi_indices)}
        
        print(f'the data length is {len(self.data_files)}, for {phase}')


    def __len__(self):
        return len(self.data_files)

    def preprocess_nifti(self, nii_path, is_img=False, vertebra=None):
        """
        Preprocessing wie in deinem Original-Dataset
        """
        nii = NII.load(nii_path, seg=not is_img)
        nii.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), 
            voxel_spacing=self.zoom, 
            verbose=False
        )
        
        if is_img:
            nii = nii.normalize_ct(min_out=0, max_out=1, inplace=True)
        
        array = nii.get_array()
        
        # Padding
        array, offset = pad_array_to_shape(array, self.input_shape)

        tensor = torch.from_numpy(array.astype(float)).unsqueeze(0)
        
        return tensor, offset#, mask

    
    def __getitem__(self, index):
        """
        Gibt einen Sample im H3DE-Net Format zurück
        """
        # Lese row aus master_df
        row = self.master_df.iloc[index]
        subject = row["subject"]
        vertebra = row["vertebra"]

        # temporär
        file_dir = row["file_dir"]

        # TODO: wie bad_pois filtern???

        if "bad_poi_list" in self.master_df.columns:
            bad_poi_list = ast.literal_eval(row["bad_poi_list"])
            bad_poi_list = [int(poi) for poi in bad_poi_list] 
        else:
            bad_poi_list = [] 

        #print(f"\nBAD POI LIST: {bad_poi_list}\n")


        
        # Lade paths
        ct_path = os.path.join(file_dir, "ct.nii.gz")
        msk_path = os.path.join(file_dir, "vertseg.nii.gz")
        subreg_path = os.path.join(file_dir, "subreg.nii.gz")
        surface_msk_path = os.path.join(file_dir, "surface_msk.nii.gz")
        poi_path = os.path.join(file_dir, self.poi_file_ending)
        
        # Get the ground truth POIs
        poi = POI.load(poi_path)
        poi.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_(
            self.zoom, verbose=False
        )

        poi_array, missing_pois = get_gt_pois(poi, vertebra, self.poi_indices)

        # Load Data
        subreg, offset = self.preprocess_nifti(subreg_path, is_img=False)
        vertseg, _ = self.preprocess_nifti(msk_path, is_img=False)

        if self.show_neighbors:
            vertseg_label = vertseg.unique()
            # Define neighbor vertebrae
            neighbor_top = vertebra - 1 if vertebra > 1 and vertebra - 1 in vertseg_label else 0  # 0 = dummy (no top/bottom neighbor)
            neighbor_bottom = vertebra + 1 if vertebra < 24 and vertebra + 1 in vertseg_label else 0

            # AUGMENTATION: Random labeldrop of neighbor vertebrae for data augmentation
            if torch.rand(1).item() < self.neighbor_drop_prob:
                if torch.rand(1).item() < 0.5:
                    # top
                    neighbor_top = 0
                else:
                    # bottom
                    neighbor_bottom = 0
        
            all_vert = [("current", vertebra), ("top", neighbor_top), ("bottom", neighbor_bottom)]

            # Filter out dummy for seg mask
            actual_vert = [vert for _, vert in all_vert if vert != 0]

            mask = np.isin(vertseg, actual_vert)
        
        else:
            mask = vertseg == vertebra
            

        # Lade Daten basierend auf input_data_type
        if self.input_data_type == "ct":

            ct, _ = self.preprocess_nifti(ct_path, is_img=True)
            input_data = ct * mask
            
        elif self.input_data_type == "subreg":

            input_data = subreg * mask
            
        elif self.input_data_type == "vertseg":

            input_data = vertseg * mask
            
        else:
            raise ValueError(f"Unknown input_data_type: {self.input_data_type}")
        
        # Adjust POIs mit offset
        poi_array = poi_array + torch.tensor(offset, dtype=torch.float32)

        max_x = self.input_shape[0] - 1
        max_y = self.input_shape[1] - 1
        max_z = self.input_shape[2] - 1
        
        outside_poi_mask = (
            (poi_array[:, 0] < 0) | (poi_array[:, 0] > max_x) |
            (poi_array[:, 1] < 0) | (poi_array[:, 1] > max_y) |
            (poi_array[:, 2] < 0) | (poi_array[:, 2] > max_z)
        )
        
        # mark outside pois with (-1000, -1000, -1000)
        poi_array[outside_poi_mask] = -1000.0

        # mark bad pois
        for bad_poi_id in bad_poi_list:
            if bad_poi_id in self.poi_idx_to_list_idx:
                list_idx = self.poi_idx_to_list_idx[bad_poi_id]
                poi_array[list_idx] = -1000.0

                #print(f"Marking bad POI ID {bad_poi_id} at list index {list_idx} as invalid.")
        
        # Mark missing POIs
        for missing_poi_id in missing_pois:
            if missing_poi_id.item() in self.poi_idx_to_list_idx:
                list_idx = self.poi_idx_to_list_idx[missing_poi_id.item()]
                poi_array[list_idx] = -1000.0

        transformed_mask = input_data > 0
        surface = compute_surface(transformed_mask, iterations=self.surface_erosion_iterations)
            
        if self.project_gt:
            # Compute surface from mask
            
            # Project POIs to surface
            poi_array, projection_distances = surface_project_coords(
                poi_array, 
                surface
            )
       


        
        _spacing = np.array(self.spacing[index])
        _filename = self.filename[index]
        
        # H3DE-Net erwartet landmarks als numpy array [N, 3]
        _img = input_data.squeeze(0).numpy()  # Remove channel dim
        _landmark = poi_array.numpy()


        # ← NEU: Stelle sicher dass surface 3D ist
        #print(f"Original surface shape: {surface.shape}")
        if surface.ndim == 4:
            surface = surface.squeeze(0)  # (1, H, W, D) → (H, W, D)

            #print(f"Surface squeezed to shape: {surface.shape}")
        elif surface.ndim == 2:
            raise ValueError(f"Surface is 2D but should be 3D: {surface.shape}")
        _surface = surface.numpy()
        #print(f"Sample image shape: {_img.shape}")

        print(f"DEBUG: dataloader-offset: {offset}")
        
        sample = {
            'image': _img,
            'landmarks': _landmark,
            'spacing': _spacing,
            'filename': _filename,
            'surface': _surface,
            'metadata': {
                'subject': subject,
                'vertebra': vertebra,
                'offset':  np.array(offset, dtype=np.float32),
                'poi_path': poi_path,
            }

        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    