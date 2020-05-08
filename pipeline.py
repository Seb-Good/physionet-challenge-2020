# import
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, GroupKFold,StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

from model.config_wavenet import hparams
from model.wavenet import DL_model
from data_generator import DataGenerator
from config import *


def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class Pipeline:
    def __init__(
        self, get_data,start_fold, epochs, batch_size, lr, pic_folder=PIC_FOLDER, debug_folder=DEBUG_FOLDER,
    ):

        # load the model
        self.start_fold = start_fold

        hparams['batch_size'] = batch_size
        hparams['epochs'] = epochs
        hparams['lr'] = lr

        print('\n')
        print('Selected Learning rate:', hparams['lr'])
        print('\n')

        self.get_data = get_data

        self.pic_folder = pic_folder
        self.debug_folder = debug_folder

    def train(self):

        # kfold cross-validation
        #kf = KFold(hparams['n_fold'], shuffle=True, random_state=42)

        kf = StratifiedKFold(hparams['n_fold'], shuffle=True, random_state=42)

        # kf = GroupKFold(n_splits=hparams['n_fold'])
        #
        group = np.arange(self.get_data.X_train.shape[0])
        step = 500000
        for i in range(int(self.get_data.X_train.shape[0]/step)):
            group[i*step:(i+1)*step] = i

        score = 0
        for fold, (train_ind, val_ind) in enumerate(
            kf.split(X=self.get_data.X_train[:,:,0],y=self.get_data.y_train[:,0,0],groups=group)
        ):

            if fold != self.start_fold:
                continue

            X_train, y_train, X_val, y_val = self.get_data.get_train_val(train_ind, val_ind)

            self.model = DL_model(n_channels=X_train.shape[2])

            # train model
            history = self.model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

            # get model predictions
            pred_val = self.model.predict(X_val)
            pred_val = self.predictions_postprocess(pred_val)
            fold_score = self.metrics(y_val, pred_val)

            # save the model
            self.model.model_save(
                hparams['model_path']
                + hparams['model_name']
                + '_'
                + str(fold)
                + '_fold_'
                + str(fold_score)
                + '.pt'
            )

            # save the sequience for debugging
            debugging = np.concatenate(
                (X_val, np.reshape(pred_val, [pred_val.shape[0], pred_val.shape[1], 1]),), axis=2,
            )
            debugging = np.concatenate((debugging, y_val), axis=2)
            np.save(self.debug_folder + str(fold) + '_fold_' + str(fold_score) + '.npy', debugging)

            # save learning curves plots
            fig, ax = plt.subplots(figsize=[10, 10])
            ax.plot(history['train_metric'])
            ax.plot(history['val_metric'])
            ax.legend(['train_f1', 'val_f1'])
            fig.savefig(self.pic_folder + 'f1_' + str(fold) + '.png')

            fig, ax = plt.subplots(figsize=[10, 10])
            ax.plot(history['train_loss'])
            ax.plot(history['val_loss'])
            ax.legend(['train_loss', 'val_loss'])
            fig.savefig(self.pic_folder + 'loss_' + str(fold) + '.png')

        return fold_score

    def predictions_postprocess(self, pred):
        return np.argmax(pred, axis=2)

    def metrics(self, y_true, pred):
        y_true = np.reshape(y_true, (-1))
        pred = np.reshape(pred, (-1))
        return f1_score(y_true, pred, average='macro', labels=list(range(11)))
