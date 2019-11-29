from torchvision.models import resnet18
import torch.nn as nn
import torch

model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

class res18(nn.Module):
    def __init__(self, num_classes=2):
        super(res18, self).__init__()
        self.base = model
        self.feature = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4          #输出512通道
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #(batch, 512, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #(batch, 512, 1, 1)
        self.reduce_layer = nn.Conv2d(1024, 512, 1)
        self.fc  = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
            )
    def forward(self, x):
        bs = x.shape[0]   #batch size
        x = self.feature(x)    # 输出512通道
        avgpool_x = self.avg_pool(x)   #输出(batch, 512, 1, 1)
        maxpool_x = self.max_pool(x)    #输出(batch,512, 1, 1)
        x = torch.cat([avgpool_x, maxpool_x], dim=1)  #输出(batch, 1024, 1, 1)
        x = self.reduce_layer(x).view(bs, -1)    #输出[batch, 512])
        logits = self.fc(x)    #输出（batch，num_classes)
        return logits