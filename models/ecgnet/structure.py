import torch
import torch.nn as nn
from loss_functions import AngularPenaltySMLoss


class Stem_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, drop_rate, pool_size):
        super().__init__()
        dilation = 1
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=pool_size, stride=2)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.drop(x)
        return x

class Stem_layer_upsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, drop_rate, scale_factor):
        super().__init__()
        dilation = 1
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2)+1,
            dilation=dilation,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.drop(x)
        return x


class Wave_block(nn.Module):
    def __init__(self, out_ch, kernel_size, dilation,drop_rate):
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

        self.conv_res = nn.Conv1d(out_ch, out_ch, 1, padding=0, dilation=dilation, bias=False,)

        self.conv_skip = nn.Conv1d(out_ch, out_ch, 1, padding=0, dilation=dilation, bias=False,)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):

        res_x = x

        tanh = self.tanh(self.bn1(self.conv1(x)))
        sig = self.sigmoid(self.bn2(self.conv2(x)))
        res = torch.mul(tanh, sig)

        res_out = self.conv_res(res) + res_x
        skip_out = self.conv_skip(res)
        return res_out, skip_out





class ECGNet(nn.Module):
    def __init__(self, n_channels, hparams, input_block=Stem_layer, basic_block=Wave_block,decoder_out_block = Stem_layer_upsample):
        super().__init__()

        self.basic_block = basic_block

        self.hparams = hparams['model']

        # stem layers
        self.layer1 = input_block(
            n_channels, self.hparams['n_filt_stem'], self.hparams['kern_size'], self.hparams['dropout'], 3
        )
        self.layer2 = input_block(
            self.hparams['n_filt_stem'],
            self.hparams['n_filt_res'],
            self.hparams['kern_size'],
            self.hparams['dropout'],
            3,
        )

        # wavenet(residual) layers
        self.layer3 = self.basic_block(self.hparams['n_filt_res'], self.hparams['kern_size'], 2,self.hparams['dropout'])
        self.layer4 = self.basic_block(self.hparams['n_filt_res'], self.hparams['kern_size'], 4,self.hparams['dropout'])
        self.layer5 = self.basic_block(self.hparams['n_filt_res'], self.hparams['kern_size'], 8,self.hparams['dropout'])
        self.layer6 = self.basic_block(self.hparams['n_filt_res'], self.hparams['kern_size'], 16,self.hparams['dropout'])
        self.layer7 = self.basic_block(self.hparams['n_filt_res'], self.hparams['kern_size'], 32,self.hparams['dropout'])
        self.layer8 = self.basic_block(self.hparams['n_filt_res'], self.hparams['kern_size'], 64,self.hparams['dropout'])
        self.layer9 = self.basic_block(self.hparams['n_filt_res'], self.hparams['kern_size'], 128,self.hparams['dropout'])
        self.layer10 = self.basic_block(self.hparams['n_filt_res'], self.hparams['kern_size'], 256,self.hparams['dropout'])


        self.conv_out_1 = input_block(
            self.hparams['n_filt_res'], self.hparams['n_filt_out_conv_1'], self.hparams['kern_size'], self.hparams['dropout'], 3
        )

        #self.bn1 = nn.BatchNorm1d(self.hparams['n_filt_out_conv_1'])

        self.conv_out_2 = input_block(
            self.hparams['n_filt_out_conv_1'], self.hparams['n_filt_out_conv_2'], self.hparams['kern_size'], self.hparams['dropout'], 2
        )




        #main head
        self.fc = nn.Linear(self.hparams['n_filt_out_conv_2'], 27)  #4733,27)#
        self.out = torch.nn.Sigmoid()

        #autoencoder head
        self.output_decoder_1 = decoder_out_block(self.hparams['n_filt_res'],self.hparams['n_filt_stem'],self.hparams['kern_size'],self.hparams['dropout'],
            2)
        self.output_decoder_2 = decoder_out_block(self.hparams['n_filt_stem'], n_channels,
                                                  1, self.hparams['dropout'],
                                                  2)




    def _make_layers(self, out_ch, kernel_size, n, basic_block):
        # dilation_rates = [2 ** i for i in range(n)]
        layers = []
        for layer in range(n):
            layers.append(basic_block(out_ch, out_ch, kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):

        # x, h_0 = self.input_layer_1(x)
        # x = x.cpu().detach()

        # x,(h_0,c_0) = self.input_layer_1(x)

        x = x.permute(0, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)

        x, skip_1 = self.layer3(x)
        #x = self.bn3(x)
        x, skip_2 = self.layer4(x)
        #x = self.bn4(x)
        x, skip_3 = self.layer5(x)
        #x = self.bn5(x)
        x, skip_4 = self.layer6(x)
        #x = self.bn6(x)
        x, skip_5 = self.layer7(x)
        #x = self.bn7(x)
        x, skip_6 = self.layer8(x)
        #x = self.bn8(x)
        x, skip_7 = self.layer9(x)
        #x = self.bn9(x)
        x, skip_8 = self.layer10(x)
        #x = self.bn10(x)




        #decoder head
        decoder_out = torch.relu(self.output_decoder_1(x))
        decoder_out = self.output_decoder_2(decoder_out)
        decoder_out = decoder_out[:,:,:-2]
        decoder_out = decoder_out.reshape(-1, decoder_out.shape[2], decoder_out.shape[1])

        #main head
        x = skip_1 + skip_2 + skip_3 + skip_4 + skip_5 + skip_6 + skip_7 + skip_8

        x = self.conv_out_1(x)
        x = self.conv_out_2(x)

        x = torch.mean(x, dim=2)


        x = self.out(self.fc(x))

        return x,decoder_out
