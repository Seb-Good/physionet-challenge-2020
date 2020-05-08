import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from pytorch_toolbelt import losses as L
from torch.utils.data import Dataset, DataLoader
from pytorchtools import EarlyStopping

# import EarlyStopping
from model.config_wavenet import hparams
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import gc
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
        #self.bn4 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 11) #

        self.main_head = self._make_layers(128, 11, 1, 1)
        self.angular_output = AngularPenaltySMLoss(in_features=128,out_features=11,loss_type='arcface')

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
        #x = self.bn1(x)
        x = self.layer2(x)
        #x = self.bn2(x)
        x = self.layer3(x)
        #x = self.bn3(x)
        x = self.layer4(x)
        #x = self.bn4(x)

        part_head = self.part_head(x)

        x = x.permute(0, 2, 1)
        part_head = part_head.permute(0, 2, 1)

        x = self.fc(x)
        part_head = self.fc(part_head)

        return x, part_head


class Dataset_train(Dataset):
    def __init__(self, input, output, output_part):
        self.input = input
        self.output = output
        self.output_part = output_part

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        x = self.input[idx]
        y = self.output[idx]
        y_part = self.output_part[idx]

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        y_part = torch.tensor(y_part, dtype=torch.float)

        return x, y, y_part


class Dataset_test(Dataset):
    def __init__(self, input):
        self.input = input

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        x = self.input[idx]
        x = torch.tensor(x, dtype=torch.float)
        return x


class DL_model:
    def __init__(self, n_channels):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # define the model
        self.model = WaveNet(n_channels=n_channels).cuda()
        summary(self.model, (hparams['model']['input_size'], n_channels))

        ########################## compile the model ###############################

        # define optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=hparams['lr'], weight_decay=1e-5
        )
        #test
        # {0: 0.005667983023907451,
        #  1: 0.0318489224862232,
        #  2: 0.059184860721374306,
        #  3: 0.05115955747500444,
        #  4: 0.08582807091547462,
        #  5: 0.12348720145657063,
        #  10: 1.0,
        #  9: 0.25507909000405593,
        #  8: 0.14042423627321626,
        #  7: 0.13084182853252133,
        #  6: 0.18530522593951732}

        #train
        # {0: 0.02482742807911037,
        #  1: 0.033233229695531795,
        #  3: 0.03960287404284465,
        #  2: 0.046687955179144924,
        #  10: 1.0,
        #  9: 0.2625110196885101,
        #  8: 0.14574012064457978,
        #  7: 0.13483387732769844,
        #  6: 0.18995598366930339,
        #  5: 0.10732016446568177,
        #  4: 0.06941722016407742}

        #weights = torch.Tensor([0.025,0.033,0.039,0.046,0.069,0.107,0.189,0.134,0.145,0.262,1]).cuda()
        self.loss_1 = nn.CrossEntropyLoss(weight=None)  # main loss
        self.loss_2 = nn.L1Loss()  # loss for the second head

        # define early stopping
        self.early_stopping = EarlyStopping(
            checkpoint_path=hparams['checkpoint_path'] + '/checkpoint.pt',
            patience=hparams['patience'],
            delta=hparams['min_delta'],
        )
        # lr cheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='max',
            factor=0.2,
            patience=3,
            verbose=True,
            threshold=hparams['min_delta'],
            threshold_mode='abs',
            cooldown=0,
            eps=0,
        )

    def fit(self, X_train, y_train, X_val, y_val):

        # history
        history_train_loss = []
        history_val_loss = []
        history_train_metric = []
        history_val_metric = []

        # calculate an average over each class for the second head
        y_partions = np.zeros((y_train.shape[0], 1, 11))
        for i in range(y_partions.shape[0]):
            for j in range(11):
                y_partions[i, 0, j] = np.where(y_train[i, :, 0] == j)[0].shape[0] / 4000

        train = Dataset_train(X_train, y_train, y_partions)
        valid = Dataset_train(X_val, y_val, y_partions)

        train_loader = torch.utils.data.DataLoader(train, batch_size=hparams['batch_size'], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=hparams['batch_size'], shuffle=True)

        for epoch in range(hparams['n_epochs']):

            # trian the model
            self.model.train()
            avg_loss = 0.0

            train_preds, train_true = torch.Tensor([]), torch.Tensor([])

            for (X_batch, y_batch, y_part_batch) in tqdm(train_loader):

                y_batch = y_batch.cuda()
                X_batch = X_batch.cuda()
                y_part_batch = y_part_batch.cuda()

                self.optimizer.zero_grad()
                # get model predictions
                pred, pred_part = self.model(X_batch)

                # process loss_1
                #pred_ = pred.view(-1, pred.shape[-1])
                pred_ = pred.reshape(-1, pred.shape[-1])
                y_batch_ = y_batch.view(-1)
                train_loss = self.loss_1(pred_, y_batch_.long())

                # process loss_2
                pred_part = pred_part.view(-1)
                y_part_batch = y_part_batch.view(-1)
                loss_2 = self.loss_2(pred_part, y_part_batch)

                if epoch <10:
                    # calculate a sum of losses:
                    train_loss += +0.5 * loss_2

                train_loss.backward()
                self.optimizer.step()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)

                # calc metric
                avg_loss += train_loss.item() / len(train_loader)

                train_true = torch.cat([train_true, y_batch_.cpu().detach()], 0)
                train_preds = torch.cat([train_preds, pred_.cpu().detach()], 0)

                X_batch = X_batch.cpu().detach()
                y_batch = y_batch.cpu().detach()
                y_part_batch = y_part_batch.cpu().detach()

            # calc triaing metric
            train_metric = f1_score(
                train_true.numpy(), train_preds.numpy().argmax(1), average='macro', labels=list(range(11))
            )

            # evaluate the model
            self.model.eval()
            val_preds, val_true = torch.Tensor([]), torch.Tensor([])
            print('Model evaluation...')
            avg_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch, y_part_batch in valid_loader:

                    y_batch = y_batch.cuda()
                    X_batch = X_batch.cuda()

                    pred, pred_partion = self.model(X_batch)
                    #pred_ = pred.view(-1, pred.shape[-1])
                    pred_ = pred.reshape(-1, pred.shape[-1])
                    y_batch_ = y_batch.view(-1)

                    avg_val_loss += self.loss_1(pred_, y_batch_.long()).item() / len(valid_loader)

                    X_batch = X_batch.cpu().detach()
                    y_batch = y_batch.cpu().detach()

                    val_true = torch.cat([val_true, y_batch_.cpu().detach()], 0)
                    val_preds = torch.cat([val_preds, pred_.cpu().detach()], 0)

            # evalueate metric
            val_metric = f1_score(
                val_true.numpy(), val_preds.numpy().argmax(1), labels=list(range(11)), average='macro'
            )

            self.scheduler.step(val_metric)
            res = self.early_stopping(score=val_metric, model=self.model)

            # print statistics
            if hparams['verbose_train']:
                print(
                    '| Epoch: ',
                    epoch + 1,
                    '| Train_loss: ',
                    avg_loss,
                    '| Train_Metrics: ',
                    train_metric,
                    '| Val_loss: ',
                    avg_val_loss,
                    '| Val_Metrics: ',
                    val_metric,
                    '| LR: ',
                    self.get_lr(self.optimizer),
                )

            # # add history
            history_train_loss.append(avg_loss)
            history_val_loss.append(avg_val_loss)
            history_train_metric.append(train_metric)
            history_val_metric.append(val_metric)

            if res == 2:
                print("Early Stopping")
                print(f'folder %d global best val max f1 model score {self.early_stopping.best_score}')
                break
            elif res == 1:
                print(f'save folder %d global val max f1 model score {val_metric}')

        history = {}
        history['train_loss'] = history_train_loss
        history['val_loss'] = history_val_loss
        history['train_metric'] = history_train_metric
        history['val_metric'] = history_val_metric

        return history

    # prediction function
    def predict(self, X_test):

        # evaluate the model
        self.model.eval()

        X_test = Dataset_test(X_test)
        test_loader = torch.utils.data.DataLoader(X_test, batch_size=hparams['batch_size'], shuffle=False)

        test_preds = torch.Tensor([])
        with torch.no_grad():
            for i, (X_batch) in enumerate(test_loader):

                X_batch = X_batch.cuda()

                pred, pred_part = self.model(X_batch)
                # pred = pred.view(-1, pred.shape[-1])

                X_batch = X_batch.cpu().detach()

                test_preds = torch.cat([test_preds, pred.cpu().detach()], 0)

                #test_preds = F.softmax(test_preds,dim=2)

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
