# dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F

class MRIDataset(Dataset):
    def __init__(self, root_dir, labels_files, phase, views= ('coronal_reg', 'axial_reg', 'sagittal_reg'), transform=None, target_size=(32, 128, 128)):
        """
        Args:
            root_dir (str): Directory with all the MRI images.
            labels_file (str): Path to the CSV file with labels.
            phase (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (tuple): Desired output size (C, D, H, W).
        """
        self.root_dir = os.path.join(root_dir, phase)
        self.phase = phase
        self.views = views
        self.transform = transform
        self.target_size = target_size
        
        self.index_offset = 1130 if phase == 'valid' else 0
        self.view_prefix = {
            'coronal_reg': 'CORO',
            'axial_reg': 'AXIA',
            'sagittal_reg': 'SAGI'
        }

        if phase == 'train':
            self.labels_abnormal = pd.read_csv(labels_files['abnormal'], header=0)
        else:
            self.labels_abnormal = pd.read_csv(labels_files['abnormal'], header=None, names=['image_name', 'Label'])

        self.labels_acl = pd.read_csv(labels_files['acl'], header=None, names=['image_name', 'Label'])
        self.labels_meniscus = pd.read_csv(labels_files['meniscus'], header=None, names=['image_name', 'Label'])

        self.labels_abnormal.sort_values('image_name', inplace=True)
        self.labels_acl.sort_values('image_name', inplace=True)
        self.labels_meniscus.sort_values('image_name', inplace=True)
        self.labels_abnormal.reset_index(drop=True, inplace=True)
        self.labels_acl.reset_index(drop=True, inplace=True)
        self.labels_meniscus.reset_index(drop=True, inplace=True)

        #self.max_samples = 10

    def __len__(self):
        return len(self.labels_abnormal)

    def __getitem__(self, idx):
        actual_idx = idx
        image_idx = actual_idx + self.index_offset

        combined_images = []
        for view in self.views:
            view_dir = os.path.join(self.root_dir, view)
            prefix = self.view_prefix[view]
            image_name = f"res_{prefix}_000_{image_idx:04d}.nii.gz"
            image_path = os.path.join(view_dir, image_name)

            if not os.path.exists(image_path):
                print(f"Image {image_path} not found!")
                return torch.zeros(3, *self.target_size), torch.zeros(3)

            try:
                image = nib.load(image_path).get_fdata()
                #print(f"[DEBUG] Loaded image for view {view}: {image.shape}")
                image = torch.tensor(image).unsqueeze(0).float()
                image = F.interpolate(image.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)
                #print(f"[DEBUG] Resized image for view {view}: {image.shape}")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return torch.zeros(3, *self.target_size), torch.zeros(3)

            combined_images.append(image)

        combined_images = torch.cat(combined_images, dim=0)  # Shape: (3 * C, D, H, W)
        #print(f"[DEBUG] Combined image shape: {combined_images.shape}")
        label_abnormal = float(self.labels_abnormal.iloc[actual_idx]['Label'])
        label_acl = float(self.labels_acl.iloc[actual_idx]['Label'])
        label_meniscus = float(self.labels_meniscus.iloc[actual_idx]['Label'])

        labels = torch.tensor([label_abnormal, label_acl, label_meniscus], dtype=torch.float32)
        #print(f"[DEBUG] Labels for idx {idx}: {labels.tolist()}")

        return combined_images, labels

class CombinedMRIFeatureDataset(Dataset):
    """
    Returns (images_3d, feats_1842, labels_3).
    We assume 'mri_dataset' gives (image_3d, label_3).
    And we have a precomputed 'features_1842' in memory or loaded from disk.
    """
    def __init__(self, mri_dataset, precomputed_features, transform=None):
        super().__init__()
        self.mri_dataset = mri_dataset  # an instance of MRIDataset
        self.features = precomputed_features  # shape: [N, 1842]
        self.transform = transform

        # We assume the labels are in the mri_dataset
        # or we fetch them from the mri_dataset __getitem__.

    def __len__(self):
        return len(self.mri_dataset)

    def __getitem__(self, idx):
        images_3d, labels_3 = self.mri_dataset[idx]  # [3, D, H, W], [3]
        feats_1842 = self.features[idx]              # [1842]
        if self.transform:
            images_3d = self.transform(images_3d)
        return images_3d, feats_1842, labels_3