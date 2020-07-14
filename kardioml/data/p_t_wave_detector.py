"""
p_t_wave_detector.py
--------------------
This module provides a function for identifying P-waves and T-waves in an ECG signal.
By: Dmitrii Shubin, 2020
"""

# 3rd party imports
import torch
import numpy as np
from torch import nn
from scipy.signal import medfilt


class PTWaveDetection(nn.Module):

    def __init__(self):
        super().__init__()

        # moving average
        self.weights_LPF = torch.Tensor(np.array([1] * 31)).cuda()
        self.weights_LPF = self.weights_LPF.view(1, 1, self.weights_LPF.shape[0])
        self.padding_LPF = int((self.weights_LPF.shape[2] - 1) / 2)
        self.padding_LPF = torch.Tensor(np.zeros((self.padding_LPF))).cuda()

    def peakdet(self, v, delta: float):

        # list of peaks
        maxtab = []
        mintab = []

        x = np.arange(v.shape[0])

        assert delta > 0, 'Delta should be positive'

        mn = np.inf
        mx = -np.inf
        mnpos = np.nan
        mxpos = np.nan

        lookformax = True

        for i in range(v.shape[0]):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            if lookformax:
                if this < mx - delta:
                    maxtab.append([mxpos, mx])
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn + delta:
                    mintab.append([mnpos, mn])
                    mx = this
                    mxpos = x[i]
                    lookformax = True

        return maxtab, mintab

    def FIR_filt(self, input, weight,padding_vector):
        input = torch.cat((input, padding_vector), 0)
        input = torch.cat((padding_vector, input), 0)
        input = input.view(1, 1, input.shape[0])
        output = torch.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        output = output.view(output.shape[2])
        return output

    def forward(self, x):

        # LPF for p and T waves cleaning
        x_filt = self.FIR_filt(x, self.weights_LPF, self.padding_LPF)

        # Standard scaling
        x_filt = (x_filt - torch.mean(x_filt)) / torch.std(x_filt)

        return x_filt

    def run(self, X, rpeaks):

        self.eval()
        X = torch.tensor(X, dtype=torch.float)
        X = X.cuda()
        X_filt = self.forward(X)
        X = X.cpu().detach().numpy()
        X_filt = X_filt.cpu().detach().numpy()
        X_filt = X_filt - medfilt(X_filt, 501)

        twaves = []
        pwaves = []

        for i in range(len(rpeaks)):

            if i == 0:
                PR = X_filt[0:rpeaks[i]].copy()
                RT = X_filt[rpeaks[i]:int(rpeaks[i] + 3 * (rpeaks[i + 1] - rpeaks[i]) / 4)].copy()
            elif i == len(rpeaks)-1:
                PR = X_filt[int(rpeaks[i] - 1 * (rpeaks[i] - rpeaks[i - 1]) / 4):rpeaks[i]].copy()
                RT = X_filt[rpeaks[i]:].copy()
            else:
                PR = X_filt[int(rpeaks[i] - 1 * (rpeaks[i] - rpeaks[i-1]) / 4):rpeaks[i]].copy()
                RT = X_filt[rpeaks[i]:int(rpeaks[i] + 3 * (rpeaks[i+1] - rpeaks[i]) / 4)].copy()

            PR /= max(PR)
            RT /= max(RT)

            T_wave, _ = self.peakdet(RT, 1 / 6)
            P_wave, _ = self.peakdet(PR, 0.1 / 6)

            if len(T_wave) > 1:
                T_wave = T_wave[1]
                twaves.append(int(rpeaks[i] + T_wave[0]))

            if len(P_wave)>0:
                # print(len(PR))
                # print(P_wave[-1][0])
                if len(PR) - P_wave[-1][0] < 45:
                    del P_wave[-1]
                if len(P_wave) > 0:
                    P_wave = P_wave[-1]
                    if i == 0:
                        pwaves.append(int(P_wave[0]))
                    else:
                        pwaves.append(int(rpeaks[i] - 1 * (rpeaks[i] - rpeaks[i-1]) / 4) + int(P_wave[0]))

        return pwaves, twaves
