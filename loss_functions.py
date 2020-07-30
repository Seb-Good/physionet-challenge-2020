import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
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
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps)
                )
                + self.m
            )
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(
                self.m
                * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps)
                )
            )

        excl = torch.cat(
            [torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class WeightedBCELoss(nn.Module):
    def __init__(self, weights=None, eps=1e-8):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights
        self.eps = eps

    def forward(self, output, target):

        loss = target * torch.log(output + self.eps) + (1 - target) * torch.log(1 - output + self.eps)

        if self.weights is not None:
            for i in range(loss.shape[1]):
                loss[:, i] *= self.weights[i]

        loss = torch.mean(loss, dim=1)  # here weights will be inserted
        loss = torch.neg(loss)

        return torch.mean(loss)


class WeightedGFLoss(nn.Module):
    def __init__(self, weights=None, eps=1e-8):
        super(WeightedGFLoss, self).__init__()
        self.weights = weights
        self.eps = eps

    def forward(self, y_pred, y_true):

        tp = torch.clamp(y_true * y_pred,0,1)
        fp = torch.clamp((1 - y_true) * y_pred,0,1)
        fn = torch.clamp(y_true * (1 - y_pred),0,1)

        # if self.weights is not None:
        #     for i in range(tp.shape[1]):
        #         tp[:,i] *= self.weights[i]
        #         fp[:, i] *= self.weights[i]
        #         fn[:, i] *= self.weights[i]

        tp = torch.sum(tp,dim=0)
        fp = torch.sum(fp,dim=0)
        fn = torch.sum(fn,dim=0)


        f2 = (5 * tp) / (5 * tp + fp + 4 * fn + self.eps)
        g2 = (tp) / (tp + fp + 2 * fn + self.eps)
        loss = f2*g2
        loss = 1 + torch.neg(loss)
        return torch.mean(loss)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
