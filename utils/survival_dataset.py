import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .survival_utils import discretize_survival_times, fit_discrete_time_cutpoints


class SurvivalWSIDataset(Dataset):
    """WSI feature dataset for discrete-time survival/prognosis tasks."""

    def __init__(
        self,
        dataset_info_csv_path,
        group,
        time_column="time_months",
        event_column="event",
        label_column="survival_label",
        num_bins=4,
        cutpoints=None,
        fit_cutpoints=False,
        max_instances=0,
        sampling="uniform",
    ):
        assert group in ["train", "val", "test"], "group must be train/val/test"
        if sampling not in {"uniform", "random", "head"}:
            raise ValueError("sampling must be uniform, random, or head")

        self.dataset_info_csv_path = dataset_info_csv_path
        self.dataset_df = pd.read_csv(dataset_info_csv_path)
        self.group = group
        self.max_instances = int(max_instances or 0)
        self.sampling = sampling

        self.slide_path_list = self._read_group_column("slide_path")
        self.times_list = [float(value) for value in self._read_group_column(time_column)]
        self.events_list = [int(value) for value in self._read_group_column(event_column)]

        prefixed_label = f"{group}_{label_column}"
        fallback_label = f"{group}_label"
        if prefixed_label in self.dataset_df or fallback_label in self.dataset_df:
            label_values = self._read_first_existing_column(
                [prefixed_label, fallback_label]
            )
            self.labels_list = [int(value) for value in label_values]
            self.cutpoints = list(cutpoints or [])
        else:
            if fit_cutpoints:
                self.cutpoints = fit_discrete_time_cutpoints(
                    self.times_list, self.events_list, num_bins=num_bins
                )
            elif cutpoints is not None:
                self.cutpoints = list(cutpoints)
            else:
                raise ValueError(
                    "cutpoints must be provided for non-training survival splits"
                )
            self.labels_list = discretize_survival_times(
                self.times_list, self.cutpoints
            ).tolist()

    def _read_group_column(self, base_name):
        return self._read_first_existing_column([f"{self.group}_{base_name}"])

    def _read_first_existing_column(self, names):
        for name in names:
            if name in self.dataset_df:
                return self.dataset_df[name].dropna().to_list()
        raise KeyError(f"Missing any of columns: {names}")

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
        feat = self._load_feature(slide_path)
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)
        indices = self._sample_indices(feat.shape[0])
        if indices is not None:
            feat = feat[torch.from_numpy(indices)]

        label = torch.tensor(int(self.labels_list[idx]), dtype=torch.long)
        event_time = torch.tensor(float(self.times_list[idx]), dtype=torch.float32)
        event = torch.tensor(int(self.events_list[idx]), dtype=torch.float32)
        return feat, label, event_time, event

    def _load_feature(self, slide_path):
        if slide_path.endswith(".h5"):
            with h5py.File(slide_path, "r") as h5_file:
                feature_dataset = h5_file["features"]
                r50_source = h5_file.attrs.get("r50_source")
                uni_source = h5_file.attrs.get("uni_source")
                if r50_source and uni_source:
                    with h5py.File(r50_source, "r") as r50_file, h5py.File(
                        uni_source, "r"
                    ) as uni_file:
                        r50_features = r50_file["features"]
                        uni_features = uni_file["features"]
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
                        feat = np.concatenate((r50_values, uni_values), axis=1)
                elif feature_dataset.ndim == 2:
                    num_instances = feature_dataset.shape[0]
                    indices = self._sample_indices(num_instances)
                    feat = feature_dataset[:] if indices is None else feature_dataset[indices]
                elif feature_dataset.ndim == 3 and feature_dataset.shape[0] == 1:
                    num_instances = feature_dataset.shape[1]
                    indices = self._sample_indices(num_instances)
                    feat = feature_dataset[0] if indices is None else feature_dataset[0, indices, :]
                else:
                    raise ValueError(
                        f"Unexpected H5 feature shape {feature_dataset.shape} "
                        f"in {slide_path}"
                    )
                return torch.from_numpy(feat)

        feat = torch.load(slide_path)
        if isinstance(feat, dict):
            if "feats" in feat:
                feat = feat["feats"]
            elif "features" in feat:
                feat = feat["features"]
            else:
                raise ValueError(
                    f"Unknown dict format in {slide_path}, keys: {list(feat.keys())}"
                )
        return feat

    def is_None_Dataset(self):
        return self.__len__() == 0

    def is_with_labels(self):
        return len(self.labels_list) != 0
