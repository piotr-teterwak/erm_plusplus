import os
import torch
from PIL import Image
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset


def metadata_values(wilds_dataset, metadata_name):
    metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
    metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
    return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSEnvironment:
    def __init__(self, wilds_dataset, metadata_name, metadata_value, transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value
        )[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


def get_camelyon(train_transform, val_transform, validation_data):

    validation_data_dict = {
        "hospital0": 0,
        "hospital1": 1,
        "hospital2": 2,
        "hospital3": 3,
        "hospital4": 4,
    }

    validation_data_list = [validation_data_dict[d] for d in validation_data]

    dataset = Camelyon17Dataset(root_dir="/projectnb/ivc-ml/piotrt/data/WILDS")
    metadata_name = "hospital"
    datasets = []
    for i, metadata_value in enumerate(metadata_values(dataset, metadata_name)):
        if i not in validation_data_list:
            env_transform = train_transform
        else:
            env_transform = val_transform

        env_dataset = WILDSEnvironment(
            dataset, metadata_name, metadata_value, env_transform
        )

        datasets.append(env_dataset)

    return datasets


def get_fmow(train_transform, val_transform, validation_data):

    validation_data_dict = {
        "region0": 0,
        "region1": 1,
        "region2": 2,
        "region3": 3,
        "region4": 4,
        "region5": 5,
    }

    validation_data_list = [validation_data_dict[d] for d in validation_data]

    dataset = FMoWDataset(root_dir="/projectnb/ivc-ml/piotrt/data/WILDS")
    metadata_name = "region"
    datasets = []
    for i, metadata_value in enumerate(metadata_values(dataset, metadata_name)):
        if i not in validation_data_list:
            env_transform = train_transform
        else:
            env_transform = val_transform

        env_dataset = WILDSEnvironment(
            dataset, metadata_name, metadata_value, env_transform
        )

        datasets.append(env_dataset)

    return datasets
