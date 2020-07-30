import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# from pytorch_toolbelt import losses as L
from pytorchtools import EarlyStopping

# import EarlyStopping
from model.config_ecgnet import hparams
import numpy as np
from tqdm import tqdm
from metrics import compute_beta_score
from loss_functions import WeightedBCELoss, WeightedGFLoss


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
    def __init__(self, in_ch, out_ch, kernel_size, dilation, pooling_factor):
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
        self.dropout = nn.Dropout(hparams['model']['dropout'])
        self.pooling = nn.MaxPool1d(kernel_size=pooling_factor)

    def forward(self, x):
        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        #x = self.dropout(x)

        x = self.pooling(x)

        return x

    def activatations(self, x):
        """

        :param x: input signal
        :return: activations of the first layer
        """
        x = self.conv(x)
        x = torch.mean(x, dim=1)
        return x


class res_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, pooling_factor):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=int(kernel_size / 2))
        self.bn_1 = nn.BatchNorm1d(out_ch)
        self.drop_1 = nn.Dropout(hparams['model']['dropout'])
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=int(kernel_size / 2))
        self.pooling_1 = nn.MaxPool1d(kernel_size=pooling_factor)

        self.conv1x1 = nn.Conv1d(in_ch, out_ch, 1, padding=0, dilation=1)
        self.pooling_2 = nn.MaxPool1d(kernel_size=pooling_factor)

        self.bn_2 = nn.BatchNorm1d(out_ch)
        self.drop_2 = nn.Dropout(hparams['model']['dropout'])

    def forward(self, res, x):

        # res = self.pooling_2(res)
        res = self.conv1x1(res)

        x = self.conv1(x)
        x = self.bn_1(x)
        x = torch.relu(x)
        x = self.drop_1(x)
        x = self.conv2(x)

        # x = self.pooling_1(x)

        res = res + x

        out2 = self.bn_2(res)
        out2 = torch.relu(out2)
        out2 = self.drop_2(out2)

        return res, out2


class EcgNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.block_length = 2400

        # channel 1
        self.layer1 = CBR(n_channels, 128, 11, 1, 2)
        self.layer2 = CBR(128, 64, 11, 1, 2)
        self.layer3 = CBR(64, 32, 11, 1, 2)
        self.layer4 = CBR(32, 16, 11, 1, 2)
        # self.layer4 = CBR(32, 16, 11, 1, 4)

        self.fc1 = nn.Linear(28800, 300)
        # self.bn_1 = nn.BatchNorm1d(300)
        self.drop_fc1 = nn.Dropout(hparams['model']['dropout'])
        self.fc2 = nn.Linear(300, 300)
        self.drop_fc2 = nn.Dropout(hparams['model']['dropout'])
        self.fc3 = nn.Linear(300, 9)

    def _make_layers(self, in_ch, out_ch, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        layers = [nn.Conv1d(in_ch, out_ch, 1)]
        for dilation in dilation_rates:
            layers.append(self.basic_block(out_ch, out_ch, kernel_size, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        cnn_out = []
        for i in range(int(x.shape[2]/self.block_length)):
            exec(f'x{i} = self.layer1(x[:,:,i*self.block_length:(i+1)*self.block_length])')
            exec(f'x{i} = self.layer2(x{i})')
            exec(f'x{i} = self.layer3(x{i})')
            exec(f'x{i} = self.layer4(x{i})')
            cnn_out.append(eval(f'x{i}'))


        x = torch.cat(cnn_out, 1)

        x = x.view(-1, int(x.shape[1] * x.shape[2]))

        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)
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
        self.loss_1 = WeightedBCELoss(weights=weights)  # nn.BCELoss()#pos_weight=None)  # main loss

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
            patience=3,
            verbose=True,
            threshold=hparams['min_delta'],
            threshold_mode='abs',
            cooldown=0,
            eps=0,
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
