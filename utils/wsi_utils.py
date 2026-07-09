import pandas as pd
import math
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


def build_wsi_datasets(args, dataset_class=None):
    dataset_class = dataset_class or WSI_Dataset
    max_instances = (
        int(args.Model.max_instances)
        if "max_instances" in args.Model
        else 0
    )
    sampling = (
        str(args.Model.sampling)
        if "sampling" in args.Model
        else "random"
    )
    path = args.Dataset.dataset_csv_path
    return (
        dataset_class(
            path,
            "train",
            max_instances=max_instances,
            sampling=sampling,
        ),
        dataset_class(
            path,
            "val",
            max_instances=max_instances,
            sampling="uniform",
        ),
        dataset_class(
            path,
            "test",
            max_instances=max_instances,
            sampling="uniform",
        ),
    )


class WSI_Dataset(Dataset):
    def __init__(
        self,
        dataset_info_csv_path,
        group,
        max_instances=0,
        sampling="uniform",
    ):
        assert group in ['train','val','test'], 'group must be in [train,val,test]'
        if sampling not in {"uniform", "random", "head"}:
            raise ValueError("sampling must be uniform, random, or head")
        self.dataset_info_csv_path = dataset_info_csv_path
        self.dataset_df = pd.read_csv(self.dataset_info_csv_path)
        self.slide_path_list = self.dataset_df[group+'_slide_path'].dropna().to_list()
        self.labels_list = self.dataset_df[group+'_label'].dropna().to_list()
        self.group = group
        self.max_instances = int(max_instances or 0)
        self.sampling = sampling

    def __len__(self):
        return len(self.slide_path_list)
    
    def _sample_indices(self, num_instances):
        if self.max_instances <= 0 or num_instances <= self.max_instances:
            return None
        if self.group == "train" and self.sampling == "random":
            indices = np.random.choice(
                num_instances, size=self.max_instances, replace=False
            )
            return np.sort(indices)
        if self.sampling == "head":
            return np.arange(self.max_instances)
        return np.linspace(
            0, num_instances - 1, self.max_instances, dtype=np.int64
        )

    def __getitem__(self, idx):

        slide_path = self.slide_path_list[idx]
        label = int(self.labels_list[idx])
        label = torch.tensor(label)

        # adapting to different feature file types(https://github.com/mahmoodlab/TRIDENT)
        if slide_path.endswith('.h5'):
            with h5py.File(slide_path, 'r') as h5_file:
                feature_dataset = h5_file['features']
                r50_source = h5_file.attrs.get('r50_source')
                uni_source = h5_file.attrs.get('uni_source')
                if r50_source and uni_source:
                    with h5py.File(r50_source, 'r') as r50_file, h5py.File(
                        uni_source, 'r'
                    ) as uni_file:
                        r50_features = r50_file['features']
                        uni_features = uni_file['features']
                        if r50_features.shape[0] != uni_features.shape[0]:
                            raise ValueError(
                                "Paired H5 feature row counts differ: "
                                f"{r50_features.shape} vs {uni_features.shape}"
                            )
                        num_instances = r50_features.shape[0]
                        indices = self._sample_indices(num_instances)
                        if indices is None:
                            r50_values = r50_features[:]
                            uni_values = uni_features[:]
                        else:
                            r50_values = r50_features[indices]
                            uni_values = uni_features[indices]
                        feat = np.concatenate(
                            (r50_values, uni_values), axis=1
                        )
                elif feature_dataset.ndim == 2:
                    num_instances = feature_dataset.shape[0]
                    indices = self._sample_indices(num_instances)
                    feat = (
                        feature_dataset[:]
                        if indices is None
                        else feature_dataset[indices]
                    )
                elif (
                    feature_dataset.ndim == 3
                    and feature_dataset.shape[0] == 1
                ):
                    num_instances = feature_dataset.shape[1]
                    indices = self._sample_indices(num_instances)
                    feat = (
                        feature_dataset[0]
                        if indices is None
                        else feature_dataset[0, indices, :]
                    )
                else:
                    raise ValueError(
                        f"Unexpected H5 feature shape "
                        f"{feature_dataset.shape} in {slide_path}"
                    )
                feat = torch.from_numpy(feat)
        else:
            feat = torch.load(slide_path)
            # Handle dictionary format (e.g., {'feats': tensor, 'coords': tensor})
            if isinstance(feat, dict):
                if 'feats' in feat:
                    feat = feat['feats']
                elif 'features' in feat:
                    feat = feat['features']
                else:
                    raise ValueError(f"Unknown dict format in {slide_path}, keys: {list(feat.keys())}")
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)
        indices = self._sample_indices(feat.shape[0])
        if indices is not None:
            feat = feat[torch.from_numpy(indices)]
        return feat,label

    def is_None_Dataset(self):
        return (self.__len__() == 0)    
    
    def is_with_labels(self):
        return (len(self.labels_list) != 0)
    
    def get_balanced_sampler(self, replacement=True):
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler

        label_counts = Counter(self.labels_list)
        weights = [1.0 / label_counts[label] for label in self.labels_list]
        num_samples = len(self.labels_list)

        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=replacement)
        return sampler


class WSI_Coord_Dataset(WSI_Dataset):
    """
    WSI dataset variant for spatial MIL models.

    If feature files contain coordinates, the returned tensor is
    ``[num_patches, feature_dim + 2]`` with ``(x, y)`` appended to the last
    dimension. Supported coordinate sources:
    - ``.h5`` files with ``features`` and ``coords`` datasets
    - ``.pt`` dicts with ``feats``/``features`` and ``coords`` keys

    If coordinates are absent, it returns features only; spatial models then use
    their pseudo-grid fallback for compatibility.
    """

    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        coords = None

        if slide_path.endswith('.h5'):
            with h5py.File(slide_path, 'r') as h5_file:
                feature_dataset = h5_file['features']
                if feature_dataset.ndim == 3:
                    feature_dataset = feature_dataset[0]
                num_instances = feature_dataset.shape[0]
                indices = self._sample_indices(num_instances)
                feat = (
                    feature_dataset[:]
                    if indices is None
                    else feature_dataset[indices]
                )
                feat = torch.from_numpy(feat)
                if 'coords' in h5_file:
                    coord_dataset = h5_file['coords']
                    if coord_dataset.ndim == 3:
                        coord_dataset = coord_dataset[0]
                    coords = torch.from_numpy(
                        coord_dataset[:]
                        if indices is None
                        else coord_dataset[indices]
                    )
        else:
            loaded = torch.load(slide_path)
            if isinstance(loaded, dict):
                if 'feats' in loaded:
                    feat = loaded['feats']
                elif 'features' in loaded:
                    feat = loaded['features']
                else:
                    raise ValueError(f"Unknown dict format in {slide_path}, keys: {list(loaded.keys())}")
                if 'coords' in loaded:
                    coords = loaded['coords']
            else:
                feat = loaded

        if len(feat.shape) == 3:
            feat = feat.squeeze(0)
        if not slide_path.endswith('.h5'):
            indices = self._sample_indices(feat.shape[0])
            if indices is not None:
                index_tensor = torch.from_numpy(indices)
                feat = feat[index_tensor]
                if coords is not None:
                    coords = coords[index_tensor]
        if coords is not None:
            if len(coords.shape) == 3:
                coords = coords.squeeze(0)
            coords = coords.to(feat.device).float()
            if coords.shape[0] == feat.shape[0] and coords.shape[-1] >= 2:
                coords = coords[:, :2]
                scale = coords.abs().amax(dim=0).clamp_min(1.0)
                coords = coords / scale
                feat = torch.cat([feat, coords], dim=-1)
            else:
                raise ValueError(
                    f"Coordinates do not match features in {slide_path}"
                )
        else:
            raise ValueError(
                f"coordinate_dim=2 requires coordinates in {slide_path}"
            )
        return feat, label

    
class CDP_MIL_WSI_Dataset(WSI_Dataset):
    def __init__(self,dataset_info_csv_path,BeyesGuassian_pt_dir,group):
        super(CDP_MIL_WSI_Dataset,self).__init__(dataset_info_csv_path,group)
        self.slide_path_list = [os.path.join(BeyesGuassian_pt_dir,os.path.basename(slide_path).replace('.pt', '_bayesian_gaussian.pt')) for slide_path in self.slide_path_list]
        

    
class LONG_MIL_WSI_Dataset(WSI_Dataset):
    def __init__(self,dataset_info_csv_path,h5_csv_path,group):
        super(LONG_MIL_WSI_Dataset,self).__init__(dataset_info_csv_path,group)
        self.h5_path_list = pd.read_csv(h5_csv_path)['h5_path'].dropna().values

    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        slide_name = os.path.basename(slide_path).replace('.pt','')
        h5_path = self._find_h5_path_by_slide_name(slide_name, self.h5_path_list)
        print(h5_path)
        h5_file = h5py.File(h5_path, 'r')
        coords = torch.from_numpy(np.array(h5_file['coords']))
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        feat = torch.load(slide_path) 
        if len(feat.shape) == 3:
            feat = feat.squeeze(0) # (N,D)
        if len(coords.shape) == 3:
            coords = coords.squeeze(0) # (N,2)
        feat_with_coords = torch.cat([feat, coords], dim=-1) # (N,D+2) 
        return feat_with_coords,label 
    
    def _find_h5_path_by_slide_name(self, slide_name, h5_paths_list):
        h5_dict = {os.path.basename(h5_path).replace('.h5', ''): h5_path for h5_path in h5_paths_list}
        return h5_dict.get(slide_name, None)

class SC_MIL_WSI_Dataset(WSI_Dataset):
    """
    Dataset for SC_MIL that can work without h5 files
    If h5_csv_path is provided, uses coords from h5 files (like LONG_MIL)
    If h5_csv_path is None, generates dummy coords based on patch indices
    """
    def __init__(self, dataset_info_csv_path, h5_csv_path=None, group='train', use_dummy_coords=True):
        super(SC_MIL_WSI_Dataset, self).__init__(dataset_info_csv_path, group)
        self.use_dummy_coords = use_dummy_coords
        
        if h5_csv_path is not None and os.path.exists(h5_csv_path):
            # Use h5 files if available
            self.h5_path_list = pd.read_csv(h5_csv_path)['h5_path'].dropna().values
            self.use_dummy_coords = False
        else:
            # Use dummy coords
            self.h5_path_list = None
            self.use_dummy_coords = True
            if h5_csv_path is not None:
                print(f"⚠️  Warning: h5_csv_path '{h5_csv_path}' not found. Using dummy coords for SC_MIL.")
    
    def _generate_dummy_coords(self, num_patches):
        """
        Generate dummy coordinates based on patch indices
        Assumes patches are arranged in a grid-like structure
        """
        # Estimate grid size (assume roughly square)
        grid_size = int(np.ceil(np.sqrt(num_patches)))
        
        # Generate 2D grid coordinates
        coords = []
        for i in range(num_patches):
            row = i // grid_size
            col = i % grid_size
            coords.append([col, row])  # (x, y) format
        
        return np.array(coords, dtype=np.float32)
    
    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        slide_name = os.path.basename(slide_path).replace('.pt', '')
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        
        # Load features
        feat = torch.load(slide_path)
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)  # (N, D)
        
        num_patches = feat.shape[0]
        
        # Get coords
        if self.use_dummy_coords:
            # Generate dummy coords
            coords_np = self._generate_dummy_coords(num_patches)
            coords = torch.from_numpy(coords_np)  # (N, 2)
        else:
            # Load from h5 file
            h5_path = self._find_h5_path_by_slide_name(slide_name, self.h5_path_list)
            if h5_path is None:
                # Fallback to dummy coords if h5 not found
                coords_np = self._generate_dummy_coords(num_patches)
                coords = torch.from_numpy(coords_np)
            else:
                h5_file = h5py.File(h5_path, 'r')
                coords = torch.from_numpy(np.array(h5_file['coords']))
                h5_file.close()
                if len(coords.shape) == 3:
                    coords = coords.squeeze(0)  # (N, 2)
        
        # Concatenate features and coords
        feat_with_coords = torch.cat([feat, coords], dim=-1)  # (N, D+2)
        return feat_with_coords, label
    
    def _find_h5_path_by_slide_name(self, slide_name, h5_paths_list):
        if h5_paths_list is None:
            return None
        h5_dict = {os.path.basename(h5_path).replace('.h5', ''): h5_path for h5_path in h5_paths_list}
        return h5_dict.get(slide_name, None)
