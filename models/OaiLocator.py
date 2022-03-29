import torch
import torch.nn as nn
import torchvision.models as models


class OaiLocator(nn.Module):
    def __init__(self, args_m):
        super(OaiLocator, self).__init__()
        self.features = getattr(models, args_m['backbone'])(pretrained=args_m['pretrained']).features
        # out channels
        if args_m['backbone'] == 'alexnet':
            fmap_c = 256
        else:
            fmap_c = 512

        self.classifier = nn.Linear(fmap_c, args_m['n_classes'])
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # (1, 3, 224, 224, 23)
        x0 = x[0]
        x0 = self.features(x0)
        x0 = self.avg(x0)[:, :, 0, 0]
        x0 = self.classifier(x0)
        return x0,