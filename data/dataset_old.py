import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image

np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, label_path, cfg, transform_train, mode="train"):
        self.cfg = cfg
        self._label_header = None
        self._image_root = label_path.split("CheXpert-v1.0-small")[0]
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [
            {"1.0": "1", "": "0", "0.0": "0", "-1.0": "0"},
            {"1.0": "1", "": "0", "0.0": "0", "-1.0": "1"},
        ]
        with open(label_path) as f:
            header = f.readline().strip("\n").split(",")
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15],
            ]
            for line in f:
                labels = []
                fields = line.strip("\n").split(",")
                image_path = fields[0]
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if (
                            self.dict[1].get(value) == "1"
                            and self.cfg.enhance_index.count(index) > 0
                        ):
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if (
                            self.dict[0].get(value) == "1"
                            and self.cfg.enhance_index.count(index) > 0
                        ):
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                image_path = os.path.join(self._image_root, image_path)
                self._image_paths.append(image_path)
                # assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                if flg_enhance and self._mode == "train":
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)
        print(self._num_image)
        self.transform_train = transform_train

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image_orig = Image.fromarray(image).convert("RGB")
        # if self._mode == 'train':
        #     image = GetTransforms(image, type=self.cfg.use_transforms_type)
        # image_orig = np.array(image)
        image = self.transform_train(image_orig)

        # print(image.size)
        # if self._mode == 'dev':
        #     image1 = self.transform_train(image_orig)
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        # labels = np.concatenate([np.array([0]),labels,np.array([0])])
        # labels = torch.from_numpy(labels)
        if self._mode == "train_finetune":
            return image, labels
        if self._mode == "val_finetune":
            return image, labels
        if self._mode == "train" or self._mode == "dev":
            # print("in train")
            image1 = self.transform_train(image_orig)
            return (image, image1, labels)
        elif self._mode == "test":
            return (image, path)
        elif self._mode == "heatmap":
            return (image, path, labels)
        else:
            raise Exception("Unknown mode : {}".format(self._mode))
