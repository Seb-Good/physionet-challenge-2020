# import
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import torch
import os
from tqdm import tqdm

from data_generator import Dataset_train, Dataset_test
from metrics import Metric
from postprocessing import PostProcessing

def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class CVPipeline:
    def __init__(self, hparams, split_table_path, split_table_name, debug_folder, model, gpu,downsample):

        # load the model

        self.hparams = hparams
        self.model = model
        self.gpu = gpu
        self.downsample = downsample

        print('\n')
        print('Selected Learning rate:', self.hparams['lr'])
        print('\n')

        self.debug_folder = debug_folder
        self.split_table_path = split_table_path
        self.split_table_name = split_table_name
        self.exclusions = []
        #     'A6181',
        #     'A5177',
        #     'A4739',
        #     'A6480',
        #     'A1670',
        #     'A2404',
        #     'A6831',
        #     'A0017',
        #     'A0750',
        #     'A2922',
        #     'A0815',
        #     'A0166',
        #     'A5285',
        #     'A1395',
        #     'A2256',
        #     'A6716',
        #     'A3546',
        #     'A5601',
        #     'A2289',
        #     'A2732',
        #     'A2482',
        #     'A4982',
        #     'A2425',
        #     'A6696',
        #     'A4756',
        #     'A0631',
        #     'A4738',
        #     'A4263',
        #     'A6216',
        #     'A6832',
        #     'A3536',
        #     'A3872',
        #     'A4218',
        #     'A5356',
        #     'A6633',
        #     'A0769',
        #     'A5618',
        #     'A2006',
        #     'A6815',
        #     'A3871',
        #     'A1579',
        #     'A2396',
        #     'A0113',
        #     'A6558',
        #     'A1332',
        #     'A0713',
        #     'A3183',
        #     'A4499',
        #     'A0080',
        #     'A4404',
        #     'A1988',
        #     'A4216',
        #     'A4318',
        #     'A1237',
        #     'A3069',
        #     'A6875',
        #     'A4552',
        #     'A0290',
        #     'A5911',
        #     'A1340',
        #     'A0649',
        #     'A3605',
        #     'A5878',
        #     'A4549',
        #     'A3087',
        #     'A5189',
        #     'A5037',
        #     'A2221',
        #     'A3985',
        #     'A5841',
        #     'A0551',
        #     'A0661',
        #     'A0145',
        #     'A2593',
        #     'A4004',
        #     'A6505',
        #     'A4652',
        #     'A5042',
        #     'A1374',
        #     'A6850',
        #     'A0441',
        #     'A4972',
        #     'A1073',
        #     'A4698',
        #     'A5488',
        #     'A4342',
        #     'A2935',
        #     'A2020',
        #     'A3170',
        #     'A0445',
        #     'A1462',
        #     'A3943',
        #     'A4315',
        #     'A0582',
        #     'A0069',
        #     'A2681',
        #     'A3025',
        #     'A3842',
        #     'A0529',
        #     'A3614',
        #     'A3314',
        #     'A6120',
        #     'A5169',
        #     'A6015',
        #     'A5331',
        #     'A2933',
        #     'A4286',
        #     'A3897',
        #     'A0848',
        #     'A0173',
        #     'A5464',
        #     'A0313',
        #     'A5347',
        #     'A4727',
        #     'A6238',
        #     'A6738',
        #     'A0018',
        #     'A2134',
        #     'A4512',
        #     'A3975',
        #     'A6429',
        #     'A5164',
        #     'A4790',
        #     'A1786',
        #     'A6290',
        #     'A1372',
        #     'A1389',
        #     'A1663',
        #     'A2743',
        #     'A1258',
        #     'A1267',
        #     'A2834',
        #     'A6099',
        #     'A4170',
        #     'A2225',
        #     'A4550',
        #     'A3561',
        #     'A4672',
        #     'A3433',
        #     'A0739',
        #     'A0055',
        #     'A3898',
        #     'A6421',
        #     'A3203',
        #     'A3066',
        #     'A1211',
        #     'A4016',
        #     'A0639',
        #     'A4685',
        #     'A6857',
        #     'A6007',
        #     'A4949',
        #     'A2529',
        #     'A6071',
        #     'A1263',
        #     'A4560',
        #     'A4119',
        #     'A5733',
        #     'A2502',
        #     'A1986',
        #     'A4996',
        #     'A5477',
        #     'A1990',
        #     'A4654',
        #     'A0865',
        #     'A2926',
        #     'A5094',
        #     'A3268',
        #     'A3666',
        #     'A6306',
        #     'A5412',
        #     'A4619',
        #     'A3172',
        #     'A0183',
        #     'A4080',
        #     'Q1325',
        #     'Q2610',
        #     'Q2866',
        #     'Q0423',
        #     'Q2949',
        #     'Q2790',
        #     'Q2811',
        #     'Q1510',
        #     'Q0549',
        #     'Q3075',
        #     'Q1691',
        #     'Q3250',
        #     'Q3279',
        #     'Q2357',
        #     'Q0312',
        #     'Q2534',
        #     'Q0830',
        #     'Q0619',
        #     'Q3040',
        #     'Q0510',
        #     'Q3266',
        #     'Q2292',
        #     'Q2888',
        #     'Q0335',
        #     'Q1017',
        #     'Q2323',
        #     'Q3559',
        #     'Q3391',
        #     'Q3496',
        #     'Q1424',
        #     'Q1379',
        #     'Q1847',
        #     'Q2822',
        #     'Q0017',
        #     'Q0329',
        #     'Q2717',
        #     'Q2215',
        #     'Q1695',
        #     'Q1217',
        #     'Q1196',
        #     'Q3396',
        #     'Q1894',
        #     'Q0134',
        #     'Q0859',
        #     'Q2177',
        #     'Q3567',
        #     'Q2979',
        #     'Q0060',
        #     'Q1083',
        #     'Q2535',
        #     'Q0257',
        #     'Q0357',
        #     'Q1152',
        #     'Q0995',
        #     'Q1734',
        #     'Q1367',
        #     'Q2570',
        #     'Q2716',
        #     'Q0386',
        #     'Q3188',
        #     'Q0151',
        #     'Q0039',
        #     'Q3343',
        #     'Q0343',
        #     'Q1671',
        #     'Q2045',
        #     'Q0022',
        #     'S0380',
        #     'S0427',
        #     'S0330',
        #     'S0400',
        #     'S0483',
        #     'S0521',
        #     'S0409',
        #     'S0487',
        #     'S0505',
        #     'S0477',
        #     'S0527',
        #     'S0485',
        #     'S0484',
        #     'S0522',
        #     'S0392',
        #     'S0453',
        #     'S0508',
        #     'S0499',
        #     'S0318',
        #     'S0317',
        #     'S0490',
        #     'S0432',
        #     'S0388',
        #     'S0495',
        #     'S0336',
        #     'S0533',
        #     'S0536',
        #     'S0385',
        #     'S0337',
        #     'S0331',
        #     'S0346',
        #     'S0475',
        #     'S0479',
        #     'S0535',
        #     'S0473',
        #     'S0528',
        #     'S0458',
        #     'S0381',
        #     'S0405',
        #     'S0472',
        #     'S0411',
        #     'S0486',
        #     'S0468',
        #     'S0364',
        #     'S0481',
        #     'S0404',
        #     'S0390',
        #     'S0469',
        #     'S0498',
        #     'S0496',
        #     'S0401',
        #     'S0491',
        #     'S0327',
        #     'S0326',
        #     'S0414',
        #     'S0391',
        #     'S0369',
        #     'S0402',
        #     'S0393',
        #     'S0403',
        #     'S0494',
        #     'S0488',
        #     'S0482',
        #     'S0370',
        #     'S0509',
        #     'S0431',
        #     'S0526',
        #     'S0471',
        #     'S0387',
        #     'S0386',
        #     'S0476',
        #     'S0497',
        #     'S0394',
        #     'S0529',
        #     'S0406',
        #     'S0463',
        #     'S0500',
        #     'S0480',
        #     'S0489',
        #     'S0371',
        #     'S0478',
        #     'S0538',
        #     'S0470',
        #     'S0332',
        #     'S0447',
        #     'S0367',
        #     'S0511',
        #     'S0382',
        #     'S0412',
        #     'S0474',
        #     'S0384',
        #     'S0534',
        #     'S0502',
        #     'S0365',
        #     'S0512',
        #     'S0464',
        #     'S0428',
        # ]

        self.splits = self.load_split_table()
        self.metric = Metric()


    def load_split_table(self):

        splits = []

        split_files = [i for i in os.listdir(self.split_table_path) if i.find('fold') != -1]

        for i in range(len(split_files)):
            data = json.load(open(self.split_table_path + str(i) + '_' + self.split_table_name))

            train_data = data['train']
            for index, i in enumerate(train_data):
                i = i.split('\\')
                i = i[-1]
                train_data[index] = i

            val_data = data['val']
            for index, i in enumerate(val_data):
                i = i.split('\\')
                i = i[-1]
                val_data[index] = i

            dataset_train = []
            for i in train_data:
                if i in self.exclusions:
                    continue
                if i[0] != 'Q' and i[0] != 'S' and i[0] != 'A' and i[0] != 'H' and i[0] != 'E':  # A, B , D, E datasets
                    continue
                dataset_train.append(i)

            dataset_val = []
            for i in val_data:
                if i in self.exclusions:
                    continue
                if i[0] != 'Q' and i[0] != 'S' and i[0] != 'A' and i[0] != 'H' and i[0] != 'E':  # A, B , D, E datasets
                    continue
                dataset_val.append(i)

            data['train'] = dataset_train
            data['val'] = dataset_val

            splits.append(data)

        splits = pd.DataFrame(splits)

        return splits

    def train(self):

        score = 0
        for fold in range(self.splits.shape[0]):

            if fold is not None:
                if fold != self.hparams['start_fold']:
                    continue
            #TODO
            train = Dataset_train(self.splits['train'].values[fold][:10], aug=False,downsample=self.downsample)
            valid = Dataset_train(self.splits['val'].values[fold][:10], aug=False,downsample=self.downsample)

            X, y = train.__getitem__(0)

            self.model = self.model(
                input_size=X.shape[0], n_channels=X.shape[1], hparams=self.hparams, gpu=self.gpu
            )

            # train model
            self.model.fit(train=train, valid=valid)

            # get model predictions
            #valid = Dataset_train(self.splits['val'].values[fold][:10], aug=False,downsample=self.downsample)
            y_val,pred_val = self.model.predict(valid)
            pred_val = np.round(pred_val,4)
            #print(pred_val)
            self.postprocessing = PostProcessing(fold=self.hparams['start_fold'])
            #print(self.postprocessing.threshold)#must be initialized before usage because the threshold is updated in .fit pipeline
            pred_val_processed = self.postprocessing.run(pred_val)
            #print(pred_val_processed)
            # TODO: add activations
            # heatmap = self.model.get_heatmap(valid)

            #y_val = valid.get_labels(self.splits['val'].values[fold][:10])
            print(y_val)
            fold_score = self.metric.compute(y_val, pred_val_processed)

            # save the model
            self.model.model_save(
                self.hparams['model_path']
                + self.hparams['model_name']+f"_{self.hparams['start_fold']}"
                + '_fold_'
                + str(fold_score)
                + '.pt'
            )


            # create a dictionary for debugging
            self.save_debug_data(pred_val, self.splits['val'].values[fold][:10])



        return fold_score

    def save_debug_data(self, pred_val, validation_list):

        for index, data in enumerate(validation_list):

            if data[0] == 'A':
                data_folder = 'A'

            elif data[0] == 'Q':
                data_folder = 'B'

            elif data[0] == 'I':
                data_folder = 'C'

            elif data[0] == 'S':
                data_folder = 'D'

            elif data[0] == 'H':
                data_folder = 'E'

            elif data[0] == 'E':
                data_folder = 'F'

            data_folder = f'./data/CV_debug/{data_folder}/'

            prediction = {}
            prediction['predicted_label'] = pred_val[index].tolist()
            # save debug data
            with open(data_folder + data + '.json', 'w') as outfile:
                json.dump(prediction, outfile)

        return True
