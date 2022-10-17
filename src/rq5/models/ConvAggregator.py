from transformers import AutoModel
import torch.nn as nn
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ConvAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_vector_size =100
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(2,8), stride=(1, 4))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(4,8), stride=(2, 4))

        self.conv2 = nn.Conv2d(4, 1, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.head = nn.Linear(100, 2)

    def forward(self, x):
        x_2 = torch.zeros([1, 1, self.max_vector_size, 768])
        x_2 = x_2.to(x[0].get_device())
        xx = x.movedim(0,-2)
        _, _, h, w = xx.shape
        if h > self.max_vector_size:
            xx = xx[:,:,0:self.max_vector_size,:]
            h = self.max_vector_size
        x_2[0, 0, 0:h, 0:w] = xx

        x_3 = self.conv1(x_2)
        x_4 = self.relu1(x_3)
        x_5 = self.pool1(x_4)
        x_6 = self.conv2(x_5)
        x_7 = self.relu2(x_6)
        x_8 = self.pool2(x_7)
        x_9 = torch.flatten(x_8, start_dim=1, end_dim=3)

        x_10 = self.head(x_9)
        return x_10