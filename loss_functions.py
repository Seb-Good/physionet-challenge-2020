import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

class AngularPenaltySMLoss(nn.Module):
    
    def __init__(
        self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None,
    ):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(
                torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps,)
                )
                + self.m
            )
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(
                self.m
                * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps,)
                )
            )

        excl = torch.cat(
            [torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class CompLoss(nn.Module):

    def __init__(self,device):
        super().__init__()
        self.weights_matrix = pd.read_csv('./metrics/weights.csv').values[:,1:]
        self.weights_matrix = torch.Tensor(self.weights_matrix).to(device)
        self.device = device

    def forward(self,pred, target):

        pred = (pred - 0.5)*2
        target = (target - 0.5)*2

        # matrix for ideal prediction
        matrix_ideal = torch.mm(target.t(), target)
        matrix_ideal = torch.abs(matrix_ideal)
        matrix_ideal = torch.matmul(matrix_ideal, self.weights_matrix)
        matrix_ideal = torch.sum(matrix_ideal)

        #matrix for predictions
        matrix = torch.mm(target.t(), pred)
        matrix = torch.abs(matrix)
        matrix = torch.matmul(matrix,self.weights_matrix)
        matrix = torch.sum(matrix)



        # matrix for prediction of only normal labels
        normal_predictions = torch.Tensor(np.zeros((target.shape[0],target.shape[1]))).to(self.device)
        normal_predictions[:,21] = 1
        matrix_norm = torch.mm(target.t(), normal_predictions)
        matrix_norm = torch.abs(matrix_norm)
        #matrix_norm = matrix_norm / torch.sum(matrix_norm)
        matrix_norm = torch.matmul(matrix_norm, self.weights_matrix)
        matrix_norm = torch.sum(matrix_norm)

        norm = torch.sum(matrix_ideal)
        norm = norm + torch.sum(matrix)

        loss = (matrix/norm - matrix_norm/norm) / (matrix_ideal/norm - matrix_norm/norm)

        return 2-loss

