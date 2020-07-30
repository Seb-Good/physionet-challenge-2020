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
            out_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class WaveNet(nn.Module):
    def __init__(self, n_channels, basic_block=wave_block):
        super().__init__()
        self.basic_block = basic_block
        self.layer1 = self._make_layers(n_channels, 16, 3, 12)
        self.bn1 = nn.BatchNorm1d(16)
        self.layer2 = self._make_layers(16, 32, 3, 8)
        self.bn2 = nn.BatchNorm1d(32)
        self.layer3 = self._make_layers(32, 64, 3, 4)
        self.bn3 = nn.BatchNorm1d(64)
        self.layer4 = self._make_layers(64, 128, 3, 1)
        # self.bn4 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 11)  #

        self.main_head = self._make_layers(128, 11, 1, 1)
        self.angular_output = AngularPenaltySMLoss(in_features=128, out_features=11, loss_type='arcface')

        self.part_head = nn.Linear(4000, 1)

    def _make_layers(self, in_ch, out_ch, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        layers = [nn.Conv1d(in_ch, out_ch, 1)]
        for dilation in dilation_rates:
            layers.append(self.basic_block(out_ch, out_ch, kernel_size, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer1(x)
        # x = self.bn1(x)
        x = self.layer2(x)
        # x = self.bn2(x)
        x = self.layer3(x)
        # x = self.bn3(x)
        x = self.layer4(x)
        # x = self.bn4(x)

        part_head = self.part_head(x)

        x = x.permute(0, 2, 1)
        part_head = part_head.permute(0, 2, 1)

        x = self.fc(x)
        part_head = self.fc(part_head)

        return x, part_head
