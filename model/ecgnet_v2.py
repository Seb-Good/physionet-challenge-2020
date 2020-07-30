import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# from pytorch_toolbelt import losses as L
from pytorchtools import EarlyStopping

# import EarlyStopping
from model.config_ecgnet_v2 import hparams
import numpy as np
from tqdm import tqdm
from metrics import compute_beta_score
from loss_functions import WeightedBCELoss, WeightedGFLoss,DiceLoss


class Block_A(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size,stride):
        super().__init__()

        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=int(kernel_size / 2),stride=stride)
        self.activation = nn.ELU()
        self.drop = nn.Dropout(hparams['model']['dropout'])

    def forward(self, x):

        x = self.conv(x)
        x = self.activation(x)
        x = self.drop(x)

        return x


class Block_B(nn.Module):
    def __init__(self, kernel_size,input_size):
        super().__init__()
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.av_pool = nn.AvgPool1d(kernel_size)
        self.dense = nn.Linear(input_size,6)
    def forward(self, x):
        x1 = self.max_pool(x)
        x2 = self.av_pool(x)
        x = x1 - x2
        x = x.view(-1, int(x.shape[1] * x.shape[2]))
        x = self.dense(x)
        return x


class res_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()

        self.activation = nn.ELU()
        self.drop = nn.Dropout(hparams['model']['dropout'])
        self.block_A_1 = Block_A(in_ch, out_ch,kernel_size,stride=2)
        self.block_A_2 = Block_A(out_ch, out_ch, kernel_size,stride=1)
        self.block_A_3 = Block_A(out_ch, out_ch, kernel_size,stride=2)
        self.out_conv = nn.Conv1d(out_ch, out_ch, kernel_size, padding=int(kernel_size / 2))

    def forward(self, x):

        x = self.activation(x)
        x = self.drop(x)
        out_1 = self.block_A_1(x)

        x = self.block_A_2(out_1)
        out_2 = self.block_A_3(x)

        out_3 = self.out_conv(x)

        return out_1,out_2,out_3


class CNN_1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv_1 = nn.Conv1d(
            in_ch,
            3,
            4,
            padding=2,
            dilation=1,
        )
        self.conv_2 = nn.Conv1d(
            in_ch,
            3,
            4,
            padding=int(6.5),
            dilation=4,
        )
        self.conv_3 = nn.Conv1d(
            in_ch,
            3,
            4,
            padding=int(12.5),
            dilation=8,
        )
        self.conv_4 = nn.Conv1d(
            9,
            12,
            4,
            padding=int(2),
            dilation=1,
        )
        self.conv_5 = nn.Conv1d(
            12,
            12,
            4,
            padding=int(2),
            dilation=1,
            stride=2
        )


        self.block_A_1 = Block_A(12, 12, 4,stride=1)
        self.res_channel_1 = nn.MaxPool1d(2)

        self.res_2 = res_block(in_ch=12,out_ch=24,kernel_size=4)
        self.res_3 = res_block(in_ch=24, out_ch=24, kernel_size=4)
        self.res_4 = res_block(in_ch=24, out_ch=48, kernel_size=4)
        self.res_channel_2 = nn.MaxPool1d(8)
        self.ccn_match = nn.Conv1d(12,48,1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(hparams['model']['dropout'])
        self.block_A_2 = Block_A(48, 48, 4,stride=1)

        self.block_B_1 = Block_B(2,  7200)
        self.block_B_2 = Block_B(2,  3600)
        self.block_B_3 = Block_B(2,  3600)
        self.block_B_4 = Block_B(2,  1800)
        self.block_B_5 = Block_B(2,  3600)
        self.block_B_6 = Block_B(2,  1776)
        self.block_B_7 = Block_B(2,  3600)

    def input_block(self,x):

        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_2(x)

        x = torch.cat([x1[:,:,:-1], x2, x3], dim=1)
        x = self.conv_4(x)

        # input resudial nework
        res = self.res_channel_1(x)
        x = self.block_A_1(x)
        x = self.conv_5(x[:,:,:-3])
        x += res  ########################################

        return x

    def forward(self, x):

        x = self.input_block(x)

        #middle 3x residual blocks
        res = self.res_channel_2(x)
        res = self.ccn_match(res)
        out1,out2,x = self.res_2(x)
        out1 = out1[:,:,:-1]
        out2 = out2[:, :, :-2]
        out3, out4, x = self.res_3(x[:,:,:-3])
        out3 = out3[:, :, :-1]
        out4 = out4[:, :, :-2]
        out5, out6, x = self.res_4(x[:,:,:-3])
        out5 = out5[:, :, :-1]
        out6 = out6[:, :, :-2]
        x = x[:, :, :-3]
        x +=res

        x = self.activation(x)
        x = self.dropout(x)
        out7 = self.block_A_2(x)
        out7 = out7[:,:,:-1]

        out1 = self.block_B_1(out1)
        out2 = self.block_B_2(out2)
        out3 = self.block_B_3(out3)
        out4 = self.block_B_4(out4)
        out5 = self.block_B_5(out5)
        out6 = self.block_B_6(out6)
        out7 = self.block_B_7(out7)


        return torch.cat([out1,out2,out3,out4,out5,out6,out7],dim=1)

    def activatations(self, x):
        """
        :param x: input signal
        :return: activations of the first layer
        """
        x = self.conv_4(x)
        x = torch.mean(x, dim=1)
        return x




class EcgNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.block_length = 2400

        # channel 1
        self.layer1 = CNN_1D(n_channels)
        self.layer2 = nn.LSTM(input_size=12,hidden_size=12,batch_first=True,num_layers=1,bidirectional=True)


        self.drop_fc2 = nn.Dropout(hparams['model']['dropout'])
        self.fc3 = nn.Linear(24, 9)

    def _make_layers(self, in_ch, out_ch, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        layers = [nn.Conv1d(in_ch, out_ch, 1)]
        for dilation in dilation_rates:
            layers.append(self.basic_block(out_ch, out_ch, kernel_size, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        #cnn_out = []
        for index,i in enumerate(range(int(x.shape[2]/self.block_length))):
            if index == 0:
                cnn_out = self.layer1(x[:,:,i*self.block_length:(i+1)*self.block_length]).view(-1,1,42)
            else:
                temp = self.layer1(x[:,:,i*self.block_length:(i+1)*self.block_length]).view(-1,1,42)
                cnn_out = torch.cat([cnn_out,temp],dim=1)

        cnn_out = cnn_out.permute(0, 2, 1)

        x,(hidden,c) = self.layer2(cnn_out)
        #hidden = hidden.permute(1, 0, 2)

        x = torch.mean(x,dim=1).view(-1,24)

        x = self.fc3(x)

        x = torch.sigmoid(x)

        return x

    def activatations(self, x):
        """
        :param x: input signal
        :return: activations of the first layer
        """
        x = x.permute(0, 2, 1)
        x = self.layer1.activatations(x)
        return x


class DL_model:
    def __init__(self, n_channels):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # define the model
        self.model = EcgNet(n_channels=n_channels).cuda()
        summary(self.model, (hparams['model']['input_size'], n_channels))

        ########################## compile the model ###############################

        # define optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=hparams['lr'], weight_decay=1e-5
        )

        weights = torch.Tensor(
            [
                0.18018018,
                0.30470914,
                0.93220339,
                0.23965142,
                0.35714286,
                0.31428571,
                0.11847065,
                0.25316456,
                1,
            ]
        ).cuda()
        weights = weights*(1/torch.min(weights))
        self.loss_1 = nn.BCELoss() #WeightedGFLoss(weights=weights )#pos_weight=None)  # main loss

        # define early stopping
        self.early_stopping = EarlyStopping(
            checkpoint_path=hparams['checkpoint_path'] + '/checkpoint.pt',
            patience=hparams['patience'],
            delta=hparams['min_delta'],
            is_maximize=False,
        )
        # lr cheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=0.2,
            patience=2,
            verbose=True,
            threshold=hparams['min_delta'],
            threshold_mode='abs',
            cooldown=0,
            eps=0.001,
        )

    def fit(self, train, valid):

        train_loader = torch.utils.data.DataLoader(train, batch_size=hparams['batch_size'], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=hparams['batch_size'], shuffle=True)

        # tensorboard object
        writer = SummaryWriter()

        for epoch in range(hparams['n_epochs']):

            # trian the model
            self.model.train()
            avg_loss = 0.0

            train_preds, train_true = torch.Tensor([]), torch.Tensor([])

            for (X_batch, y_batch) in tqdm(train_loader):

                y_batch = y_batch.cuda()
                X_batch = X_batch.cuda()

                self.optimizer.zero_grad()
                # get model predictions
                pred = self.model(X_batch)

                # process loss_1
                pred = pred.view(-1, pred.shape[-1])
                y_batch = y_batch.view(-1, y_batch.shape[-1])
                train_loss = self.loss_1(pred, y_batch)

                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)
                self.optimizer.step()

                # calc metric
                avg_loss += train_loss.item() / len(train_loader)

                X_batch = X_batch.cpu().detach()
                y_batch = y_batch.cpu().detach()
                pred = pred.cpu().detach()

                train_true = torch.cat([train_true, y_batch], 0)
                train_preds = torch.cat([train_preds, pred], 0)

            # calc triaing metric
            train_preds = self.apply_threshold(train_preds.numpy())
            f2_train, g2_train = compute_beta_score(train_true.numpy(), train_preds, beta=2, num_classes=9)

            gm_train = np.sqrt(f2_train * g2_train)

            # evaluate the model
            self.model.eval()
            val_preds, val_true = torch.Tensor([]), torch.Tensor([])
            print('Model evaluation...')
            avg_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:

                    y_batch = y_batch.cuda()
                    X_batch = X_batch.cuda()

                    pred = self.model(X_batch)
                    pred = pred.reshape(-1, pred.shape[-1])
                    y_batch = y_batch.view(-1, y_batch.shape[-1])

                    avg_val_loss += self.loss_1(pred, y_batch).item() / len(valid_loader)

                    X_batch = X_batch.cpu().detach()
                    y_batch = y_batch.cpu().detach()
                    pred = pred.cpu().detach()

                    val_true = torch.cat([val_true, y_batch], 0)
                    val_preds = torch.cat([val_preds, pred], 0)

            # evalueate metric
            val_preds = self.apply_threshold(val_preds.numpy())
            f2_val, g2_val = compute_beta_score(val_true.numpy(), val_preds, beta=2, num_classes=9)

            gm_val = np.sqrt(f2_val * g2_val)

            self.scheduler.step(avg_val_loss)
            res = self.early_stopping(score=avg_val_loss, model=self.model)

            # print statistics
            if hparams['verbose_train']:
                print(
                    '| Epoch: ',
                    epoch + 1,
                    '| Train_loss: ',
                    avg_loss,
                    '| F2_train: ',
                    f2_train,
                    '| G2_train: ',
                    g2_train,
                    '| GM_train: ',
                    gm_train,
                    '| Val_loss: ',
                    avg_val_loss,
                    '| F2_val: ',
                    f2_val,
                    '| G2_val: ',
                    g2_val,
                    '| GM_val: ',
                    gm_val,
                    '| LR: ',
                    self.get_lr(self.optimizer),
                )

            # # add history to tensorboard
            writer.add_scalars('Loss', {'Train_loss': avg_loss, 'Val_loss': avg_val_loss}, epoch)

            writer.add_scalars('F2', {'F2_train': f2_train, 'F2_val': f2_val}, epoch)

            writer.add_scalars('G2', {'G2_train': g2_train, 'G2_val': g2_val}, epoch)

            writer.add_scalars('Geometric mean', {'GM_train': gm_train, 'GM_val': gm_val}, epoch)

            if res == 2:
                print("Early Stopping")
                print(f'global best min val_loss model score {self.early_stopping.best_score}')
                break
            elif res == 1:
                print(f'save global val_loss model score {avg_val_loss}')

        writer.close()

        return True

    # TODO
    # prediction function
    def predict(self, X_test):

        # evaluate the model
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(X_test, batch_size=hparams['batch_size'], shuffle=False)

        test_preds = torch.Tensor([])
        with torch.no_grad():
            for i, (X_batch) in enumerate(test_loader):

                X_batch = X_batch.cuda()

                pred = self.model(X_batch)

                X_batch = X_batch.cpu().detach()

                test_preds = torch.cat([test_preds, pred.cpu().detach()], 0)

        return test_preds.numpy()

    def get_heatmap(self, X_test):

        # evaluate the model
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(X_test, batch_size=hparams['batch_size'], shuffle=False)

        test_preds = torch.Tensor([])
        with torch.no_grad():
            for i, (X_batch) in enumerate(test_loader):
                X_batch = X_batch.cuda()

                pred = self.model.activatations(X_batch)
                pred = torch.sigmoid(pred)

                X_batch = X_batch.cpu().detach()

                test_preds = torch.cat([test_preds, pred.cpu().detach()], 0)

        return test_preds.numpy()

    def model_save(self, model_path):
        torch.save(self.model, model_path)
        return True

    def model_load(self, model_path):
        self.model = torch.load(model_path)
        return True

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def apply_threshold(self, y):

        y[np.where(y > 0.5)] = 1
        y[np.where(y <= 0.5)] = 0

        return y
