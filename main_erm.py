#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from data.dataset import construct_dataset
from models.resnet import resnet18, resnet50, wide_resnet50_2
from models.timm_model_wrapper import TimmWrapper
from util import MovingAvg, AverageMeter, ProgressMeter, accuracy
import timm
import numpy as np


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


model_names = ["regnet", "resnet", "resnet_timm_augmix", "resnet_timm_a1",  "meal_v2"]
dataset_names = [
    "domainnet",
    "terraincognita",
    "officehome",
    "pacs",
    "vlcs",
    "wilds_fmow",
]

parser = argparse.ArgumentParser(description="ERM++ training")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "--dataset", default="domainnet", choices=dataset_names, help="which dataset to use"
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--steps", default=5000, type=int, metavar="N", help="number of total steps to run"
)
parser.add_argument(
    "--linear-steps",
    default=-1,
    type=int,
    metavar="N",
    help="number of total steps to run",
)
parser.add_argument(
    "--sma-start-iter",
    default=100,
    type=int,
    metavar="N",
    help="Where to start model averaging.",
)
parser.add_argument(
    "--accum-iter",
    default=1,
    type=int,
    metavar="N",
    help="number of steps between updates",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=5e-5,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--miro-weight",
    default=0.1,
    type=float,
    metavar="MW",
    help="initial learning rate",
    dest="miro_weight",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0.0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--eval-path",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--training_data",
    default=["sketch", "real"],
    type=str,
    nargs="*",
    help="training subsets",
)
parser.add_argument(
    "--validation_data",
    default=["painting"],
    type=str,
    nargs="*",
    help="testing subsets",
)


parser.add_argument(
    "--save_name", default="", type=str, help="name of saved checkpoint"
)
parser.add_argument(
    "--save-dir",
    default="model_checkpoints/",
    type=str,
    help="name of saved checkpoint",
)
parser.add_argument("--freeze-bn", dest="freeze_bn", action="store_true")
parser.add_argument("--miro", dest="miro", action="store_true")
parser.add_argument(
    "--save-freq", default=-1, type=int, help="how often to save checkpoints in steps"
)
parser.add_argument(
    "--train-val-split", default=-1, type=float, help="how much to split train val"
)

parser.add_argument("--pretrained", dest="pretrained", action="store_true")
parser.add_argument("--sma", dest="sma", action="store_true")
parser.set_defaults(pretrained=False)
parser.set_defaults(sma=False)
parser.set_defaults(freeze_bn=False)
parser.set_defaults(miro=False)

best_acc1 = 0
best_steps = 0

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    global best_acc1
    global best_steps

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_datasets, test_dataset, num_classes = construct_dataset(
        args, train_transform, val_transform
    )

    batch_size_list = [args.batch_size] * len(train_datasets)

    train_loader = [
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        for bs, train_dataset in zip(batch_size_list, train_datasets)
    ]

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(test_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == "resnet50":
        model = resnet50(
            pretrained=args.pretrained,
            num_classes=num_classes,
            freeze_bn=args.freeze_bn,
            projection_head=args.projection_head,
        )
    elif args.arch == "resnet_timm_a1":
        model = TimmWrapper(
            timm.create_model(
                "resnet50",
                pretrained=False,
                num_classes=num_classes,
                features_only=args.miro,
            ),
            freeze_bn=args.freeze_bn,
            miro=args.miro,
            num_classes=num_classes,
        )
        state_dict = {
           "model." +  k.split("module.")[-1]: v
            for (k, v) in torch.load(
                "model_checkpoints/resnet_a1/resnet50_a1_0-14fe96d1.pth"
            ).items()
        }
        del state_dict["model.fc.weight"]
        del state_dict["model.fc.bias"]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"model.fc.weight", "model.fc.bias"}
    elif args.arch == "resnet_timm_augmix":
        model = TimmWrapper(
            timm.create_model(
                "resnet50",
                pretrained=False,
                num_classes=num_classes,
                features_only=args.miro,
            ),
            freeze_bn=args.freeze_bn,
            miro=args.miro,
            num_classes=num_classes,
        )
        state_dict = {
           "model." +  k.split("module.")[-1]: v
            for (k, v) in torch.load(
                "model_checkpoints/augmix/resnet50_ram-a26f946b.pth"
            ).items()
        }
        del state_dict["model.fc.weight"]
        del state_dict["model.fc.bias"]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"model.fc.weight", "model.fc.bias"}
    elif args.arch == "meal_v2":
        model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes)
        state_dict = {
            k.split("module.")[-1]: v
            for (k, v) in torch.load(
                "model_checkpoints/meal_v2/MEALV2_ResNet50_224.pth"
            ).items()
        }
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"model.fc.weight", "model.fc.bias"}
    else:
        raise RuntimeError("Invalid architechture specified")

    if args.miro:
        featurizer = TimmWrapper(
            timm.create_model(
                "resnet50", pretrained=True, num_classes=num_classes, features_only=True
            ),
            freeze_bn=True,
            miro=args.miro,
            num_classes=num_classes,
            freeze_all=True,
        ).to(device)
        shapes = miro_nets.get_shapes(featurizer, (3, 224, 224))
        mean_encoders = nn.ModuleList(
            [miro_nets.MeanEncoder(shape).to(device) for shape in shapes]
        )
        var_encoders = nn.ModuleList(
            [miro_nets.VarianceEncoder(shape).to(device) for shape in shapes]
        )
    else:
        featurizer = None
        mean_encoders = None
        var_encoders = None

    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Num params:{}".format(params))

    model = torch.nn.DataParallel(model).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if args.miro:
        backbone_parameters = [
            {"params": model.parameters()},
            {"params": mean_encoders.parameters(), "lr": args.lr * 10},
            {"params": var_encoders.parameters(), "lr": args.lr * 10},
        ]
    else:
        backbone_parameters = model.parameters()

    optimizer = torch.optim.Adam(
        backbone_parameters, args.lr, weight_decay=args.weight_decay
    )

    linear_parameters = []

    for n, p in model.named_parameters():
        if "fc" in n:
            linear_parameters.append(p)

    linear_optimizer = torch.optim.Adam(
        linear_parameters, args.lr, weight_decay=args.weight_decay
    )

    # Load model for eval
    if args.eval_path:
        if os.path.isfile(args.eval_path):
            print("=> loading checkpoint '{}'".format(args.eval_path))
            checkpoint = torch.load(args.eval_path)
            model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.eval_path, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.eval_path))

    cudnn.benchmark = True

    model = MovingAvg(model, args.sma, args.sma_start_iter)

    if args.evaluate:
        validate(val_loader, model, criterion, args, steps)
        return

    int(args.steps / len(train_loader))
    epoch = 0
    steps = 0
    save_iterate = 0

    if args.evaluate:
       validate(val_loader, model, criterion, args, steps)
       return


    while True:
        if steps > args.steps:
            break

        steps, save_iterate = train(
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            linear_optimizer,
            epoch,
            args,
            steps,
            save_iterate,
            featurizer=featurizer,
            mean_encoders=mean_encoders,
            var_encoders=var_encoders,
        )
        epoch = epoch + 1

    acc1 = validate(val_loader, model, criterion, args, steps)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    save_name = "{}_{}".format(args.save_name, save_iterate)
    save_checkpoint(
        {
            "epoch": epoch,
            "arch": args.arch,
            "state_dict": model.network_sma.state_dict(),
            "best_acc1": best_acc1,
            "optimizer": optimizer.state_dict(),
        },
        is_best,
        save_name=save_name,
        save_dir=args.save_dir,
        save_iterate=save_iterate,
        args=args,
    )


def train(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    linear_optimizer,
    epoch,
    args,
    steps,
    save_iterate,
    featurizer,
    mean_encoders,
    var_encoders,
):
    global best_acc1
    global best_steps
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to train mode
    model.network.train()
    model.network_sma.train()

    train_loader_epoch = train_loader.copy()

    train_loader_main_idx = np.argmax([len(d) for d in train_loader_epoch])
    train_loader_main = train_loader_epoch.pop(train_loader_main_idx)

    aux_iter_list = [iter(aux_loader) for aux_loader in train_loader_epoch]
    end = time.time()

    progress = ProgressMeter(
        len(train_loader_main),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    for i, (images, target) in enumerate(train_loader_main):
        if steps > args.steps:
            return steps, save_iterate
        if steps > args.linear_steps:
            selected_optimizer = optimizer
        else:
            selected_optimizer = linear_optimizer

        steps = steps + 1
        # measure data loading time
        aux_images_list = []
        aux_target_list = []
        for idx, aux_iter in enumerate(aux_iter_list):
            try:
                aux_images, aux_target = next(aux_iter)
            except StopIteration:
                aux_iter_list[idx] = iter(train_loader_epoch[idx])
                aux_images, aux_target = next(aux_iter_list[idx])
            aux_images_list.append(aux_images)
            aux_target_list.append(aux_target)

        images = images.to(device, non_blocking=True)
        for idx in range(len(aux_iter_list)):
            aux_images_list[idx] = aux_images_list[idx].to(device, non_blocking=True)
        target = target.cuda(non_blocking=True)
        for idx in range(len(aux_iter_list)):
            aux_target_list[idx] = aux_target_list[idx].to(device, non_blocking=True)

        images = torch.concat([images] + aux_images_list, dim=0)
        target = torch.concat([target] + aux_target_list)

        data_time.update(time.time() - end)

        # compute output
        if args.miro:
            with torch.no_grad():
                _, pre_feats = featurizer.forward_features(images)
            output, inter_feats = model.network.module.forward_features(images)
            loss = criterion(output, target)
            reg_loss = 0.0
            for f, pre_f, mean_enc, var_enc in zip(
                inter_feats, pre_feats, mean_encoders, var_encoders
            ):
                # mutual information regularization
                mean = mean_enc(f)
                var = var_enc(f)
                vlb = (mean - pre_f).pow(2).div(var) + var.log()
                reg_loss += vlb.mean() / 2.0

            loss += reg_loss * args.miro_weight

        else:
            output = model.network(images)
            loss = criterion(output, target)
        # measure accuracy and record loss
        max_k = 5
        acc1, acc5 = accuracy(output, target, topk=(1, max_k))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        loss = loss / args.accum_iter

        loss.backward()

        if ((i + 1) % args.accum_iter == 0) or (i + 1 == len(train_loader_main)):
            selected_optimizer.step()
            selected_optimizer.zero_grad()
            model.update_sma()
        if not args.freeze_bn:
            model.network_sma(images)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if args.save_freq > 0 and (steps % args.save_freq == 0):

            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args,steps)
            # switch to train mode
            model.network.train()
            model.network_sma.train()

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_steps = steps
                save_checkpoint(
                     {
                         "epoch": epoch,
                         "arch": args.arch,
                         "state_dict": model.network_sma.state_dict(),
                         "best_acc1": best_acc1,
                         "optimizer": optimizer.state_dict(),
                     },
                     is_best,
                     save_name=args.save_name,
                     save_dir=args.save_dir,
                     args=args,
                 )


    return steps, save_iterate


def validate(val_loader, model, criterion, args, steps):
    global best_steps
    global best_acc1
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.network.eval()
    model.network_sma.eval()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model.network_sma(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        max_k = 5
        acc1, acc5 = accuracy(output, target, topk=(1, max_k))

        losses.update(loss.item(), images.size(0))

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # remember best acc@1 and save checkpoint
    is_best = top1.avg > best_acc1
    best_acc1 = max(top1.avg, best_acc1)
    if is_best:
        best_steps = steps


    progress.display_summary()
    print("Best acc steps: {}".format(best_steps))

    return top1.avg


def save_checkpoint(state, is_best, save_name, save_dir, args=None):
    filename = save_dir + "checkpoint_" + str(save_name) + ".pth.tar"
    torch.save(state, filename)
    if is_best:
        best_filename = save_dir + "model_best_" + str(args.save_name) + ".pth.tar"
        shutil.copyfile(filename, best_filename)


if __name__ == "__main__":
    main()
