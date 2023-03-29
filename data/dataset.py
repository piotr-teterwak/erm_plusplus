import copy
import os

from data.wilds_datasets import get_fmow


import torch
import torchvision.datasets as datasets


TERRAINCOGNITA_PATH = "/projectnb/ivc-ml/piotrt/data/terra_incognita/terra_incognita/"
DOMAINNET_PATH = "/projectnb/ivc-ml/piotrt/data/domainnet"
OFFICEHOME_PATH = "/projectnb/ivc-ml/piotrt/data/office_home"
PACS_PATH = "/projectnb/ivc-ml/piotrt/data/PACS"
VLCS_PATH = "/projectnb/ivc-ml/piotrt/data/VLCS"


def _construct_dataset_helper(args, dataset_dict, train_transform, val_transform):
    train_datasets = []
    val_datasets = []
    train_dataset_lengths = []
    test_dataset = None

    for d in args.training_data:
        train_datasets.append(dataset_dict[d])
        train_dataset_lengths.append(len(dataset_dict[d]))

    for d in args.validation_data:
        val_datasets.append(dataset_dict[d])

    if args.train_val_split > 0:
        datasets_split_train = []
        datasets_split_val = []
        for d in train_datasets:
            lengths = [int(len(d) * args.train_val_split)]
            lengths.append(len(d) - lengths[0])
            train_split, val_split = torch.utils.data.random_split(
                d, lengths, torch.Generator().manual_seed(42)
            )
            train_split.dataset = copy.copy(d)
            train_split.dataset.transform = train_transform
            datasets_split_train.append(train_split)
        for idx, d in enumerate(val_datasets):
            lengths = [int(len(d) * args.train_val_split)]
            lengths.append(len(d) - lengths[0])
            train_split, val_split = torch.utils.data.random_split(
                 d, lengths, torch.Generator().manual_seed(42)
            )
            val_split.dataset.transform = val_transform
            datasets_split_val.append(val_split)

        train_datasets = datasets_split_train
        test_dataset = datasets_split_val

    else:

        test_dataset = val_datasets

    return train_datasets, test_dataset


def construct_dataset(args, train_transform, val_transform):

    if args.dataset == "domainnet":

        num_classes = 345

        transform_dict = {
            "sketch": train_transform,
            "real": train_transform,
            "clipart": train_transform,
            "infograph": train_transform,
            "quickdraw": train_transform,
            "painting": train_transform,
        }

        for d in args.validation_data:
            transform_dict[d] = val_transform

        sketch_dataset = datasets.ImageFolder(
            os.path.join(DOMAINNET_PATH, "sketch"),
            transform=transform_dict["sketch"],
        )
        real_dataset = datasets.ImageFolder(
            os.path.join(DOMAINNET_PATH, "real"),
            transform=transform_dict["real"],
        )
        clipart_dataset = datasets.ImageFolder(
            os.path.join(DOMAINNET_PATH, "clipart"),
            transform=transform_dict["clipart"],
        )
        infograph_dataset = datasets.ImageFolder(
            os.path.join(DOMAINNET_PATH, "infograph"),
            transform=transform_dict["infograph"],
        )
        quickdraw_dataset = datasets.ImageFolder(
            os.path.join(DOMAINNET_PATH, "quickdraw"),
            transform=transform_dict["quickdraw"],
        )
        painting_dataset = datasets.ImageFolder(
            os.path.join(DOMAINNET_PATH, "painting"),
            transform=transform_dict["painting"],
        )

        dataset_dict = {
            "sketch": sketch_dataset,
            "real": real_dataset,
            "clipart": clipart_dataset,
            "infograph": infograph_dataset,
            "quickdraw": quickdraw_dataset,
            "painting": painting_dataset,
        }

    elif args.dataset == "terraincognita":

        num_classes = 10

        transform_dict = {
            "location_100": train_transform,
            "location_38": train_transform,
            "location_43": train_transform,
            "location_46": train_transform,
        }

        for d in args.validation_data:
            transform_dict[d] = val_transform

        location_100_dataset = datasets.ImageFolder(
            os.path.join(TERRAINCOGNITA_PATH, "location_100"),
            transform=transform_dict["location_100"],
        )
        location_38_dataset = datasets.ImageFolder(
            os.path.join(TERRAINCOGNITA_PATH, "location_38"),
            transform=transform_dict["location_38"],
        )
        location_43_dataset = datasets.ImageFolder(
            os.path.join(TERRAINCOGNITA_PATH, "location_43"),
            transform=transform_dict["location_43"],
        )
        location_46_dataset = datasets.ImageFolder(
            os.path.join(TERRAINCOGNITA_PATH, "location_46"),
            transform=transform_dict["location_46"],
        )

        dataset_dict = {
            "location_100": location_100_dataset,
            "location_38": location_38_dataset,
            "location_43": location_43_dataset,
            "location_46": location_46_dataset,
        }

    elif args.dataset == "officehome":

        num_classes = 65

        transform_dict = {
            "art": train_transform,
            "clipart": train_transform,
            "product": train_transform,
            "real": train_transform,
        }

        for d in args.validation_data:
            transform_dict[d] = val_transform

        art_dataset = datasets.ImageFolder(
            os.path.joing(OFFICEHOME_PATH, "Art"),
            transform=transform_dict["art"],
        )
        clipart_dataset = datasets.ImageFolder(
            os.path.joing(OFFICEHOME_PATH, "Clipart"),
            transform=transform_dict["clipart"],
        )
        product_dataset = datasets.ImageFolder(
            os.path.joing(OFFICEHOME_PATH, "Product"),
            transform=transform_dict["product"],
        )
        real_dataset = datasets.ImageFolder(
            os.path.joing(OFFICEHOME_PATH, "Real"),
            transform=transform_dict["real"],
        )

        dataset_dict = {
            "art": art_dataset,
            "clipart": clipart_dataset,
            "product": product_dataset,
            "real": real_dataset,
        }

    elif args.dataset == "pacs":

        num_classes = 7

        transform_dict = {
            "art_painting": train_transform,
            "cartoon": train_transform,
            "photo": train_transform,
            "sketch": train_transform,
        }

        for d in args.validation_data:
            transform_dict[d] = val_transform

        art_painting_dataset = datasets.ImageFolder(
            os.path.join(PACS_PATH, "art_painting"),
            transform=transform_dict["art_painting"],
        )
        cartoon_dataset = datasets.ImageFolder(
            os.path.join(PACS_PATH, "cartoon"),
            transform=transform_dict["cartoon"],
        )
        photo_dataset = datasets.ImageFolder(
            os.path.join(PACS_PATH, "photo"),
            transform=transform_dict["photo"],
        )
        sketch_dataset = datasets.ImageFolder(
            os.path.join(PACS_PATH, "sketch"),
            transform=transform_dict["sketch"],
        )

        dataset_dict = {
            "art_painting": art_painting_dataset,
            "cartoon": cartoon_dataset,
            "photo": photo_dataset,
            "sketch": sketch_dataset,
        }

    elif args.dataset == "vlcs":

        num_classes = 5

        transform_dict = {
            "caltech101": train_transform,
            "labelme": train_transform,
            "sun09": train_transform,
            "voc2007": train_transform,
        }

        for d in args.validation_data:
            transform_dict[d] = val_transform

        caltech101_dataset = datasets.ImageFolder(
            os.path.join(VLCS_PATH, "Caltech101"),
            transform=transform_dict["caltech101"],
        )
        labelme_dataset = datasets.ImageFolder(
            os.path.join(VLCS_PATH, "LabelMe"),
            transform=transform_dict["labelme"],
        )
        sun09_dataset = datasets.ImageFolder(
            os.path.join(VLCS_PATH, "SUN09"),
            transform=transform_dict["sun09"],
        )
        voc2007_dataset = datasets.ImageFolder(
            os.path.join(VLCS_PATH, "VOC2007"),
            transform=transform_dict["voc2007"],
        )

        dataset_dict = {
            "caltech101": caltech101_dataset,
            "labelme": labelme_dataset,
            "sun09": sun09_dataset,
            "voc2007": voc2007_dataset,
        }

    elif args.dataset == "wilds_fmow":

        num_classes = 62

        train_datasets = []
        val_datasets = []
        train_dataset_lengths = []
        test_dataset = None

        datasets_list = get_fmow(train_transform, val_transform, args.validation_data)

        region0_dataset = datasets_list[0]
        region1_dataset = datasets_list[1]
        region2_dataset = datasets_list[2]
        region3_dataset = datasets_list[3]
        region4_dataset = datasets_list[4]
        region5_dataset = datasets_list[5]

        dataset_dict = {
            "region0": region0_dataset,
            "region1": region1_dataset,
            "region2": region2_dataset,
            "region3": region3_dataset,
            "region4": region4_dataset,
            "region5": region5_dataset,
        }

    train_dataset, test_dataset = _construct_dataset_helper(
        args, dataset_dict, train_transform, val_transform
    )

    return train_dataset, test_dataset, num_classes
