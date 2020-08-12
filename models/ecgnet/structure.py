import torch
import torch.nn as nn
from loss_functions import AngularPenaltySMLoss


class Stem_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, drop_rate):
        super().__init__()
        dilation = 1
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
        )
        #self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=4)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.drop(x)
        return x

class Wave_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        dilation = 1
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
        )

        self.conv_res = nn.Conv1d(
            out_ch,
            out_ch,
            1,
            padding=0,
            dilation=dilation,
        )

        self.conv_sip_channel = nn.Conv1d(
            in_ch,
            out_ch,
            1,
            padding=0,
            dilation=dilation,
        )

        self.conv_skip = nn.Conv1d(
            out_ch,
            out_ch,
            1,
            padding=0,
            dilation=dilation,
        )


        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        res_x = self.conv_sip_channel(x)

        tanh = self.tanh(self.conv1(x))
        sig = self.sigmoid(self.conv2(x))
        res = tanh.mul(sig)

        res_out = self.conv_res(res)+ res_x
        skip_out = self.conv_skip(res)
        #res_out = res_out
        return res_out,skip_out


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
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x


class ECGNet(nn.Module):
    def __init__(self, n_channels, input_block=Stem_layer,basic_block=Wave_block):
        super().__init__()

        #self.input_layer_1 = nn.RNN(input_size=n_channels,hidden_size=500,num_layers=1,batch_first=True,bidirectional=False)

        #stem layers
        self.layer1 = input_block(n_channels, 32, 10,0.3)
        self.layer2 = input_block(n_channels, 64, 10,0.3)

        #wavenet(residual) layers
        self.layer3 = self.basic_block(64, 64, 10)
        self.layer4 = self.basic_block(64, 64, 10)
        self.layer5 = self.basic_block(64, 64, 10)
        self.layer6 = self.basic_block(64, 64, 10)
        self.layer7 = self.basic_block(64, 64, 10)
        self.layer8 = self.basic_block(64, 64, 10)
        self.layer9 = self.basic_block(64, 64, 10)
        self.layer10 = self.basic_block(64, 64, 10)



        self.fc1 = nn.Linear(1184, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 27)#
        self.out = torch.nn.Hardsigmoid()

    def _make_layers(self, out_ch, kernel_size, n, basic_block):
        #dilation_rates = [2 ** i for i in range(n)]
        layers = []
        for layer in range(n):
            layers.append(basic_block(out_ch, out_ch, kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):

        #x, h_0 = self.input_layer_1(x)
        #x = x.cpu().detach()

        #x,(h_0,c_0) = self.input_layer_1(x)

        x = x.permute(0,2,1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(-1,x.shape[1]*x.shape[2])


        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(self.fc3(x))

        return x
