# basic libs
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

# pytorch
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# custom modules
from metrics import Metric
from utils.torchsummary import summary
from utils.pytorchtools import EarlyStopping
from torch.nn.parallel import DataParallel as DP


# model
from models.wavenet.structure import WaveNet


class Model:
    """
    This class handles basic methods for handling the model:
    1. Fit the model
    2. Make predictions
    3. Save
    4. Load
    """

    def __init__(self, input_size, n_channels, hparams):

        self.hparams = hparams

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # define the models
        self.model = WaveNet(n_channels=n_channels).to(self.device)
        summary(self.model, (input_size, n_channels))
        # self.model.half()

        if torch.cuda.device_count() > 1:
            print("Number of GPUs will be used: ", torch.cuda.device_count() - 3)
            self.model = DP(self.model, device_ids=list(range(torch.cuda.device_count() - 3)))
        else:
            print('Only one GPU is available')

        self.metric = Metric()
        self.num_workers = 1
        ########################## compile the model ###############################

        # define optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.hparams['lr'], weight_decay=1e-5
        )

        # weights = torch.Tensor([0.025,0.033,0.039,0.046,0.069,0.107,0.189,0.134,0.145,0.262,1]).cuda()
        self.loss = nn.BCELoss()  # CompLoss(self.device)

        # define early stopping
        self.early_stopping = EarlyStopping(
            checkpoint_path=self.hparams['checkpoint_path'] + '/checkpoint.pt',
            patience=self.hparams['patience'],
            delta=self.hparams['min_delta'],
        )
        # lr cheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='max',
            factor=0.2,
            patience=3,
            verbose=True,
            threshold=self.hparams['min_delta'],
            threshold_mode='abs',
            cooldown=0,
            eps=0,
        )

        self.seed_everything(42)
        self.threshold = 0.75
        self.scaler = torch.cuda.amp.GradScaler()

    def seed_everything(self, seed):
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)

    def fit(self, train, valid):

        train_loader = DataLoader(
            train, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=self.num_workers
        )  # ,collate_fn=train.my_collate
        valid_loader = DataLoader(
            valid, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=self.num_workers
        )  # ,collate_fn=train.my_collate

        # tensorboard object
        writer = SummaryWriter()

        for epoch in range(self.hparams['n_epochs']):

            # trian the model
            self.model.train()
            avg_loss = 0.0

            train_preds, train_true = torch.Tensor([]), torch.Tensor([])

            for (X_batch, y_batch) in tqdm(train_loader):
                y_batch = y_batch.float().to(self.device)
                X_batch = X_batch.float().to(self.device)

                self.optimizer.zero_grad()
                # get model predictions
                pred = self.model(X_batch)
                X_batch = X_batch.cpu().detach()

                # process loss_1
                pred = pred.view(-1, pred.shape[-1])
                y_batch = y_batch.view(-1, y_batch.shape[-1])
                train_loss = self.loss(pred, y_batch)
                y_batch = y_batch.float().cpu().detach()
                pred = pred.float().cpu().detach()

                train_loss.backward() #self.scaler.scale(train_loss).backward()  #
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)
                self.optimizer.step() # self.scaler.step(self.optimizer)  #
                self.scaler.update()

                # calc metric
                avg_loss += train_loss.item() / len(train_loader)

                train_true = torch.cat([train_true, y_batch], 0)
                train_preds = torch.cat([train_preds, pred], 0)

            # calc triaing metric
            train_preds = train_preds.numpy()
            train_preds[np.where(train_preds >= self.threshold)] = 1
            train_preds[np.where(train_preds < self.threshold)] = 0
            metric_train = self.metric.compute(labels=train_true.numpy(), outputs=train_preds)

            # evaluate the model
            print('Model evaluation...')
            self.model.zero_grad()
            self.model.eval()
            val_preds, val_true = torch.Tensor([]), torch.Tensor([])
            avg_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    y_batch = y_batch.float().to(self.device)
                    X_batch = X_batch.float().to(self.device)

                    pred = self.model(X_batch)
                    X_batch = X_batch.float().cpu().detach()

                    pred = pred.reshape(-1, pred.shape[-1])
                    y_batch = y_batch.view(-1, y_batch.shape[-1])

                    avg_val_loss += self.loss(pred, y_batch).item() / len(valid_loader)
                    y_batch = y_batch.float().cpu().detach()
                    pred = pred.float().cpu().detach()

                    val_true = torch.cat([val_true, y_batch], 0)
                    val_preds = torch.cat([val_preds, pred], 0)

            # evalueate metric
            val_preds = val_preds.numpy()
            val_preds[np.where(val_preds >= self.threshold)] = 1
            val_preds[np.where(val_preds < self.threshold)] = 0
            metric_val = self.metric.compute(val_true.numpy(), val_preds)

            self.scheduler.step(avg_val_loss)
            res = self.early_stopping(score=avg_val_loss, model=self.model)

            # print statistics
            if self.hparams['verbose_train']:
                print(
                    '| Epoch: ',
                    epoch + 1,
                    '| Train_loss: ',
                    avg_loss,
                    '| Val_loss: ',
                    avg_val_loss,
                    '| Metric_train: ',
                    metric_train,
                    '| Metric_val: ',
                    metric_val,
                    '| Current LR: ',
                    self.__get_lr(self.optimizer),
                )

            # # add history to tensorboard
            writer.add_scalars(
                'Loss', {'Train_loss': avg_loss, 'Val_loss': avg_val_loss}, epoch,
            )

            writer.add_scalars('Metric', {'Metric_train': metric_train, 'Metric_val': metric_val}, epoch)

            if res == 2:
                print("Early Stopping")
                print(f'global best min val_loss model score {self.early_stopping.best_score}')
                break
            elif res == 1:
                print(f'save global val_loss model score {avg_val_loss}')

        writer.close()

        self.model.zero_grad()

        return True

    def predict(self, X_test):

        # evaluate the model
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(
            X_test, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=self.num_workers
        )  # ,collate_fn=train.my_collate

        test_preds = torch.Tensor([])
        print('Start generation of predictions')
        with torch.no_grad():
            for i, (X_batch, y_batch) in enumerate(tqdm(test_loader)):
                X_batch = X_batch.float().to(self.device)

                pred = self.model(X_batch)

                X_batch = X_batch.float().cpu().detach()

                test_preds = torch.cat([test_preds, pred.cpu().detach()], 0)

        return test_preds.numpy()

    def get_heatmap(self, X_test):

        # evaluate the model
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(
            X_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )  # ,collate_fn=train.my_collate

        test_preds = torch.Tensor([])
        with torch.no_grad():
            for i, (X_batch) in enumerate(test_loader):
                X_batch = X_batch.float().to(self.device)

                pred = self.model.activatations(X_batch)
                pred = torch.sigmoid(pred)

                X_batch = X_batch.float().cpu().detach()

                test_preds = torch.cat([test_preds, pred.cpu().detach()], 0)

        return test_preds.numpy()

    def model_save(self, model_path):
        torch.save(self.model, model_path)
        return True

    def model_load(self, model_path):
        self.model = torch.load(model_path)
        return True

    ################## Utils #####################

    def __get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


class CompLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weights_matrix = pd.read_csv('./metrics/weights.csv').values[:, 1:]
        self.weights_matrix = torch.Tensor(self.weights_matrix).to(device)
        self.device = device

    def forward(self, pred, target):

        pred = (pred - 0.5) * 2
        target = (target - 0.5) * 2

        # matrix for ideal prediction
        matrix_ideal = torch.mm(target.t(), target)
        matrix_ideal = torch.abs(matrix_ideal)
        matrix_ideal = torch.matmul(matrix_ideal, self.weights_matrix)
        matrix_ideal = torch.sum(matrix_ideal)

        # matrix for predictions
        matrix = torch.mm(target.t(), pred)
        matrix = torch.abs(matrix)
        matrix = torch.matmul(matrix, self.weights_matrix)
        matrix = torch.sum(matrix)

        # matrix for prediction of only normal labels
        normal_predictions = torch.Tensor(np.zeros((target.shape[0], target.shape[1]))).to(self.device)
        normal_predictions[:, 21] = 1
        matrix_norm = torch.mm(target.t(), normal_predictions)
        matrix_norm = torch.abs(matrix_norm)
        # matrix_norm = matrix_norm / torch.sum(matrix_norm)
        matrix_norm = torch.matmul(matrix_norm, self.weights_matrix)
        matrix_norm = torch.sum(matrix_norm)

        norm = torch.sum(matrix_ideal)
        norm = norm + torch.sum(matrix)

        loss = (matrix / norm - matrix_norm / norm) / (matrix_ideal / norm - matrix_norm / norm)

        return 2 - loss
