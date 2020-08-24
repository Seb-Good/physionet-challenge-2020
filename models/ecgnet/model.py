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
from loss_functions import CompLoss,FocalLoss
from postprocessing import PostProcessing
from data_generator import Preprocessing

# model
from models.ecgnet.structure import ECGNet

class Model:
    """
    This class handles basic methods for handling the model:
    1. Fit the model
    2. Make predictions
    3. Save
    4. Load
    """

    def __init__(self, input_size, n_channels, hparams, gpu,inference=False):

        self.hparams = hparams

        if inference:
            self.device = torch.device('cpu')
            self.model = ECGNet(n_channels=n_channels, hparams=self.hparams).to(self.device)
        else:
            if torch.cuda.device_count() > 1:
                if len(gpu) > 0:
                    print("Number of GPUs will be used: ", len(gpu))
                    self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                    self.model = ECGNet(n_channels=n_channels, hparams=self.hparams).to(self.device)
                    self.model = DP(self.model, device_ids=gpu,output_device=gpu[0])
                else:
                    print("Number of GPUs will be used: ", torch.cuda.device_count() - 5)
                    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    self.model = ECGNet(n_channels=n_channels, hparams=self.hparams).to(self.device)
                    self.model = DP(self.model, device_ids=list(range(torch.cuda.device_count() - 5)))
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model = ECGNet(n_channels=n_channels, hparams=self.hparams).to(self.device)
                print('Only one GPU is available')


        # define the models
        #summary(self.model, (input_size, n_channels))
        #print(torch.cuda.is_available())

        self.metric = Metric()
        self.num_workers = 18
        self.threshold = 0.5

        ########################## compile the model ###############################

        # define optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.hparams['lr'])

        weights = torch.Tensor([1. , 1. , 1. , 1. , 0.5, 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0.5,
       0.5, 1. , 1. , 1. , 1. , 0.5, 1. , 1. , 1. , 1. , 0.5, 1. , 1. ,
       0.5]).to(self.device)


        self.loss = nn.BCELoss(weight=weights)# CompLoss(self.device) #
        self.decoder_loss = nn.MSELoss()

        # define early stopping
        self.early_stopping = EarlyStopping(
            checkpoint_path=self.hparams['checkpoint_path'] + '/checkpoint'+str(self.hparams['start_fold'])+'.pt',
            patience=self.hparams['patience'],
            delta=self.hparams['min_delta'],
            is_maximize=True,
        )
        # lr cheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='max',
            factor=0.2,
            patience=1,
            verbose=True,
            threshold=self.hparams['min_delta'],
            threshold_mode='abs',
            cooldown=0,
            eps=0,
        )

        self.seed_everything(42)

        self.postprocessing = PostProcessing(fold=self.hparams['start_fold'])
        self.scaler = torch.cuda.amp.GradScaler()

    def seed_everything(self, seed):
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)

    def fit(self, train, valid):

        train_loader = DataLoader(
            train, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=self.num_workers
        )
        valid_loader = DataLoader(
            valid, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=self.num_workers
        )

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
                pred,pred_decoder = self.model(X_batch)

                # process loss_1
                pred = pred.view(-1, pred.shape[-1])
                pred = pred ** 2
                y_batch = y_batch.view(-1, y_batch.shape[-1])
                train_loss = self.loss(pred, y_batch)


                y_batch = y_batch.float().cpu().detach()
                pred = pred.float().cpu().detach()

                # process loss_2
                pred_decoder = pred_decoder.view(-1, pred_decoder.shape[-1])
                X_batch = X_batch.view(-1, X_batch.shape[-1])
                decoder_train_loss = self.decoder_loss(pred_decoder, X_batch)
                X_batch = X_batch.float().cpu().detach()
                pred_decoder = pred_decoder.float().cpu().detach()

                # calc loss
                avg_loss += train_loss.item() / len(train_loader)

                #sum up multi-head losses
                train_loss = train_loss + decoder_train_loss

                self.scaler.scale(train_loss).backward()  # train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)
                self.scaler.step(self.optimizer)  # self.optimizer.step()
                self.scaler.update()



                train_true = torch.cat([train_true, y_batch], 0)
                train_preds = torch.cat([train_preds, pred], 0)

            # calc triaing metric
            train_preds = train_preds.numpy()
            train_true = train_true.numpy()

            threshold = self.postprocessing.find_opt_thresold(train_true,train_preds)
            self.postprocessing.update_threshold(threshold)
            train_preds = self.postprocessing.run(train_preds)
            metric_train = self.metric.compute(labels=train_true, outputs=train_preds)

            # evaluate the model
            print('Model evaluation...')
            self.model.eval()
            val_preds, val_true = torch.Tensor([]), torch.Tensor([])
            avg_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    y_batch = y_batch.float().to(self.device)
                    X_batch = X_batch.float().to(self.device)

                    pred,pred_decoder = self.model(X_batch)
                    pred_decoder = pred_decoder.float().cpu().detach()
                    X_batch = X_batch.float().cpu().detach()

                    pred = pred.reshape(-1, pred.shape[-1])
                    pred = pred ** 2
                    y_batch = y_batch.view(-1, y_batch.shape[-1])

                    avg_val_loss += self.loss(pred, y_batch).item() / len(valid_loader)
                    y_batch = y_batch.float().cpu().detach()
                    pred = pred.float().cpu().detach()



                    val_true = torch.cat([val_true, y_batch], 0)
                    val_preds = torch.cat([val_preds, pred], 0)

            # evalueate metric
            val_preds = val_preds.numpy()
            val_true = val_true.numpy()
            # val_true, val_preds = self.metric.find_opt_thresold(val_true, val_preds)
            val_preds = self.postprocessing.run(val_preds)
            metric_val = self.metric.compute(val_true, val_preds)

            self.scheduler.step(metric_val)#avg_val_loss)
            res = self.early_stopping(score=metric_val, model=self.model,threshold=threshold)

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
                print(f'global best max val_loss model score {self.early_stopping.best_score}')
                break
            elif res == 1:
                print(f'save global val_loss model score {metric_val}')

        writer.close()

        self.model = self.early_stopping.load_best_weights()
        self.postprocessing.update_threshold(self.early_stopping.threshold)

        return True

    def predict(self, X_test):

        # evaluate the model
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(
            X_test, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=self.num_workers
        )  # ,collate_fn=train.my_collate

        test_preds = torch.Tensor([])
        test_val = torch.Tensor([])
        print('Start generation of predictions')
        with torch.no_grad():
            for i, (X_batch, y_batch) in enumerate(tqdm(test_loader)):
                X_batch = X_batch.float().to(self.device)

                pred,pred_decoder = self.model(X_batch)
                pred = pred ** 2

                X_batch = X_batch.float().cpu().detach()

                test_preds = torch.cat([test_preds, pred.cpu().detach()], 0)
                test_val = torch.cat([test_val, y_batch.cpu().detach()], 0)

        return test_val.numpy(),test_preds.numpy()

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
                pred = pred ** 2
                X_batch = X_batch.float().cpu().detach()

                test_preds = torch.cat([test_preds, pred.cpu().detach()], 0)

        return test_preds.numpy()

    def model_save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        # self.model.module.state_dict(), PATH
        # torch.save(self.model, model_path)
        return True

    def model_load(self, model_path):
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        return True

    def model_load_old(self, model_path):
        self.model = torch.load(model_path,map_location=self.device)
        return True

    def inference(self,X,y):


        preprocessing = Preprocessing(aug=False)

        X = preprocessing.run(X,y,label_process=False)




        X = X.reshape(1,-1,X.shape[1])

        self.model.eval()
        predictions,pred_decoder = self.model.forward(torch.Tensor(X))
        predictions = predictions **2
        predictions = predictions.detach().numpy()
        print(np.round(predictions,3))


        return predictions

    ################## Utils #####################

    def __get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
