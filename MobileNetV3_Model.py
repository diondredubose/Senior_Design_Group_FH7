import torch
import torch.nn as nn
import frame_generator as fg
import numpy as np

class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.sigmoid(x + 3) / 6

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0):
        super(MobileNetV3, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.width_mult = width_mult

        self.features = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
            # Depthwise convolution layer 1
            nn.Conv2d(16, 16, kernel_size=3, stride=1, groups=16, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
            # Pointwise convolution layer 1
            nn.Conv2d(16, 24, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(24),
            hswish(),
            # Depthwise convolution layer 2
            nn.Conv2d(24, 24, kernel_size=3, stride=2, groups=24, padding=1, bias=False),
            nn.BatchNorm2d(24),
            hswish(),
            # Pointwise convolution layer 2
            nn.Conv2d(24, 40, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(40),
            hswish(),
            # ...Additional layers of depthwise and pointwise convolution
        )
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x: torch.tensor):
        print(x.shape)
        print(x.dtype)
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(-1, 1280)
        x = self.classifier(x)
        return x

ECJ = fg.AgentDataset("roomrecordings_2023_01_22.zip", transform=None)
image, depth = ECJ.__getitem__(0)

Backbone = MobileNetV3()
y = Backbone.forward(torch.tensor(torch.from_numpy(np.array(image)).unsqueeze(0).permute(0,3,1,2)).byte())
h=0
