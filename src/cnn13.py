import torch
import torch.nn as nn


class CNN13(nn.Module):
    """
    CNN-13 is a network with 10 convolutional layers follower by 3 linear layers for classification.
    """
    def __init__(
            self, 
            in_channels: int = 3, 
            num_classes: int = 100
        ):
        super().__init__()

        self.features = nn.Sequential(
            # input shape (3, 32, 32)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), # 3 
            nn.ReLU(inplace=True),

            # input shape (64, 32, 32)
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32 -> 16
            nn.Dropout(0.1),

            # input shape (64, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # input shape (128, 16, 16)
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16 -> 8
            nn.Dropout(0.2),

            # input shape (128, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # input shape (256, 8, 8)
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8 -> 4
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            # input shape (256, 4, 4)
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)         
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out