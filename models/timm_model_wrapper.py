import timm
import torch
import torch.nn as nn


class TimmWrapper(nn.Module):
    def __init__(
        self, model, freeze_bn=False, miro=False, num_classes=100, freeze_all=False
    ):
        super(TimmWrapper, self).__init__()
        self.model = model
        self.freeze_bn = freeze_bn
        self.miro = miro
        self.num_classes = num_classes
        self.freeze_all = freeze_all

        if self.freeze_all:
            for p in model.parameters():
                p.requires_grad_(False)

        if self.freeze_bn:
            self.freeze_batchnorm()

        if self.miro:
            self.global_pool, self.fc = timm.models.resnet.create_classifier(
                timm.models.resnet.Bottleneck.expansion * 512, self.num_classes, "avg"
            )

    def forward(self, x):
        if self.miro:
            x = self.model(x)[-1]
            x = self.global_pool(x)
            return self.fc(x)
        else:
            return self.model(x)

    def forward_features(self, x):
        if self.miro:
            x = self.model(x)
            x_pool = self.global_pool(x[-1])
            out = self.fc(x_pool)
            x[0] = self.model.maxpool(x[0])
            return (out, x)
        else:
            return self.model(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_bn:
            self.freeze_batchnorm()

    def freeze_batchnorm(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
