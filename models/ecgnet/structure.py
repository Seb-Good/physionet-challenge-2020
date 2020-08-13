import torch
import torch.nn as nn
from loss_functions import AngularPenaltySMLoss


class Stem_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, drop_rate,pool_size):
        super().__init__()
        dilation = 1
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            stride=1,
            bias = False,
        )
        #self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=pool_size,stride=2)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.drop(x)
        return x

class Wave_block(nn.Module):
    def __init__(self, out_ch, kernel_size,dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_ch = out_ch

        self.conv1 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            bias=False,
        )

        self.conv_res = nn.Conv1d(
            out_ch,
            out_ch,
            1,
            padding=0,
            dilation=dilation,
            bias=False,
        )


        self.conv_skip = nn.Conv1d(
            out_ch,
            out_ch,
            1,
            padding=0,
            dilation=dilation,
            bias=False,
        )


        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        res_x = x

        tanh = self.tanh(self.conv1(x))
        sig = self.sigmoid(self.conv2(x))
        res = torch.mul(tanh,sig)

        res_out = self.conv_res(res)+ res_x
        skip_out = self.conv_skip(res)
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
        self.basic_block = basic_block

        #stem layers
        self.layer1 = input_block(n_channels, 32, 9,0.3,3)
        self.layer2 = input_block(32, 64, 9,0.3,3)

        #wavenet(residual) layers
        self.layer3 = self.basic_block(64, 9,2)
        self.layer4 = self.basic_block(64, 9,4)
        self.layer5 = self.basic_block(64, 9,8)
        self.layer6 = self.basic_block(64, 9,16)
        self.layer7 = self.basic_block(64, 9,32)
        self.layer8 = self.basic_block(64, 9,64)
        self.layer9 = self.basic_block(64, 9,128)
        self.layer10 = self.basic_block(64, 9,256)


        self.conv_out_1 = self.conv2 = nn.Conv1d(
            64,
            128,
            9,
            padding=int((10 + (10 - 1) * (0 - 1)) / 2),
            dilation=1,
            bias=False,
        )

        self.conv_out_2 = self.conv2 = nn.Conv1d(
            128,
            256,
            9,
            padding=int((10 + (10 - 1) * (0 - 1)) / 2),
            dilation=1,
            bias=False,
        )



        self.fc = nn.Linear(256, 27)#
        self.out = torch.nn.Sigmoid()

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

        x, skip_1 = self.layer3(x)
        x, skip_2 = self.layer4(x)
        x, skip_3 = self.layer5(x)
        x, skip_4 = self.layer6(x)
        x, skip_5 = self.layer7(x)
        x, skip_6 = self.layer8(x)
        x, skip_7 = self.layer9(x)
        x, skip_8 = self.layer10(x)

        x = skip_1 + skip_2 + skip_3 + skip_4 + skip_5 + skip_6 + skip_7 + skip_8

        x = self.conv_out_1(x)
        x = self.conv_out_2(x)

        x = torch.mean(x,dim=2)


        x = self.out(self.fc(x))

        return x
