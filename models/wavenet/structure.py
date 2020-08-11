import torch
import torch.nn as nn
from loss_functions import AngularPenaltySMLoss


class wave_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
        )
        self.conv3 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
        )
        self.conv4 = nn.Conv1d(out_ch, out_ch, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res_x = x
        tanh = self.tanh(self.conv2(x))
        sig = self.sigmoid(self.conv3(x))
        res = tanh.mul(sig)
        x = self.conv4(res)
        x = res_x + x
        return x


class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
        )
        #self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x


class WaveNet(nn.Module):
    def __init__(self, n_channels, basic_block=CBR):
        super().__init__()

        self.input_layer_1 = nn.GRU(input_size=n_channels,hidden_size=500,num_layers=1,batch_first=True,bidirectional=False)

        self.basic_block = basic_block
        self.layer1 = self.basic_block(1, 128, 3, 12)
        self.layer2 = self.basic_block(128, 64, 3, 8)
        self.layer3 = self.basic_block(64, 32, 3, 4)

        self.fc1 = nn.Linear(1984, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 27)#


    def _make_layers(self, in_ch, out_ch, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        layers = [nn.Conv1d(in_ch, out_ch, 1)]
        for dilation in dilation_rates:
            layers.append(self.basic_block(out_ch, out_ch, kernel_size, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):

        x, h_0 = self.input_layer_1(x)
        #x,(h_0,c_0) = self.input_layer_1(x)

        h_0 = h_0.permute(1,0,2)
        x = self.layer1(h_0)
        x = self.layer2(x)
        x = self.layer3(x)


        x = x.view(-1,x.shape[1]*x.shape[2])


        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x
