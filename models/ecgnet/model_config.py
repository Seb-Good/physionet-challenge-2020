import os

hparams = {}
# training params
hparams['n_epochs'] = 1
hparams['lr'] = 0.001
hparams['batch_size'] = 3
hparams['verbose_train'] = True

# early stopping settings
hparams['min_delta'] = 0.001  # thresold of improvement
hparams['patience'] = 10  # wait for n epoches for emprovement
hparams['n_fold'] = 5  # number of folds for cross-validation
hparams['verbose'] = True  # print score or not
hparams['start_fold'] = 1

# directories
hparams['model_path'] = './data/model_weights'
hparams['model_path'] += '/ecgnet_model'
hparams['checkpoint_path'] = hparams['model_path'] + '/checkpoint'
hparams['model_name'] = '/ecgnet'

for path in [hparams['model_path'], hparams['checkpoint_path']]:
    os.makedirs(path, exist_ok=True)

# dictionary of hyperparameters
structure_hparams = dict()
# global dropout rate
structure_hparams['dropout'] = 0.3
# number of filers for the models
structure_hparams['kern_size'] = 9
structure_hparams['n_filt_stem'] = 32
structure_hparams['n_filt_res'] = 64
structure_hparams['n_filt_out_conv_1'] = 128
structure_hparams['n_filt_out_conv_2'] = 256

hparams['model'] = structure_hparams
