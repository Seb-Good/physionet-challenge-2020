# import libs
import torch
from torch.utils.data import Dataset

import numpy as np

np.random.seed(42)

# DATA GENERATOR
class DataGenerator:
    """
    this if the data generator class for classic machine leanring algorithms (XGBoost, LGBM, etc.)
    """

    def __init__(
        self,
        category=CAT,
        train_name=TRAIN_NAME,
        test_name=TEST_NAME,
        data_folder=DATA_FOLDER,
        target=TARGET,
        droplist=DROPLIST,
        columns=COLUMNS,
    ):

        self.category = category
        self.train_name = train_name
        self.test_name = test_name
        self.data_folder = data_folder
        self.target = target
        self.droplist = droplist
        self.columns = columns

        self.train = self.load_data()

    # initial preprocessing
    def preprocessing(self, train):

        if len(self.columns) > 0:
            train = train[self.columns + self.target]

        if len(self.droplist) > 0:
            train = train.drop(self.droplist, axis=1)

        # get columns with nans
        nan_feat = train.columns[train.isna().any()].tolist()

        self.nan_feature = []

        for i in nan_feat:
            train[i + '_nan'] = 0
            train[i + '_nan'][train[i].isna()] = 1
            self.category += [i + '_nan']
            self.nan_feature += [i]

        # train, self.category = self.cat_permutations(train, self.category)

        train = reduce_mem_usage(train)

        return train

    def get_train_val(self, train_ind, val_ind, droplist):

        df_train = self.train.iloc[train_ind, :]
        df_val = self.train.iloc[val_ind, :]

        df_train, df_val, target_train, target_val = self.preprocessing_train(df_train, df_val, droplist)

        return df_train, df_val, target_train, target_val

    # partial data load
    def part_data(self, column):

        # train set
        train = pd.read_csv(self.data_folder + self.train_name, usecols=column, index_col=None)

        # test set
        test = pd.read_csv(self.data_folder + self.test_name, usecols=column, index_col=None)

        for f in column:
            if f in self.category:
                le = LabelEncoder()

                temp = train.append(test, ignore_index=True)
                temp[f] = le.fit_transform(list(temp[f].values))

                train = temp.iloc[: train.shape[0], :]
                test = temp.iloc[train.shape[0] :, :]

        return train, test

    # initial data load
    def load_data(self):

        # train set
        train = pd.read_csv(self.data_folder + self.train_name, index_col=0)

        # apply preprocessing:
        train = self.preprocessing(train)

        return train

    # load test data
    def load_data_test(self, droplist):

        # test set
        test = pd.read_csv(self.data_folder + self.test_name, index_col=0)

        ID_column = test['TransactionID'].values

        # apply preprocessing:
        test = self.preprocessing_test(test, droplist)

        return test, ID_column

    def preprocessing_test(self, df_test, droplist):

        if len(self.droplist) > 0:
            df_test = df_test.drop(self.droplist, axis=1)

        # get columns with nans
        for i in self.nan_feature:
            df_test[i + '_nan'] = 0
            df_test[i + '_nan'][df_test[i].isna()] = 1

        # train, self.category = self.cat_permutations(train, self.category)

        train = reduce_mem_usage(df_test)

        df_test = df_test.drop('TransactionID', axis=1)

        # drop features from the droplist
        if len(droplist) > 0:
            df_test = df_test.drop(droplist, axis=1)

        cat_features = [value for value in df_test.columns.tolist() if value in self.category]

        # make target encoding
        pred_enc(
            df=df_test,
            dictionary=self.dictionary,
            cat_features=cat_features,
            target_col=self.target,
            glob_mean=self.glob_mean,
        )

        # drop target column
        df_test = df_test.drop(self.target[0], axis=1)

        return df_test

    def preprocessing_train(self, df_train, df_val, droplist):

        # drop features from the droplist
        if len(droplist) > 0:
            df_train = df_train.drop(droplist, axis=1)
            df_val = df_val.drop(droplist, axis=1)

        cat_features = [value for value in df_train.columns.tolist() if value in self.category]

        # make target encoding
        self.alpha = create_alpha(cat_features)
        df_val, self.glob_mean, self.dictionary = test_enc(
            df_train, df_val, cat_features=cat_features, target_col=self.target, alpha=self.alpha,
        )

        df_train = train_enc(
            df_train,
            cat_features=cat_features,
            target_col=self.target,
            alpha=self.alpha,
            glob_mean=self.glob_mean,
        )

        # get target columns
        target_train = df_train[self.target[0]]
        target_val = df_val[self.target[0]]

        df_train = df_train.drop(self.target[0], axis=1)
        df_val = df_val.drop(self.target[0], axis=1)

        # apply upsampling
        temp = df_train.columns.tolist()
        df_train, target_train = self.upsampling(df_train.values, target_train.values)
        df_train = pd.DataFrame(df_train, columns=temp)

        return df_train, df_val, target_train, target_val

    # upsampling to make class balancing equal
    def upsampling(self, X_loc, y_loc):

        UniqClass = np.unique(y_loc)

        mostQreq = 0
        numSam_max = 0

        for i in UniqClass:
            numSam = np.where(y_loc == i)[0].shape[0]
            if numSam_max < numSam:
                numSam_max = numSam
                mostQreq = i

        for i in UniqClass:
            if i == mostQreq:
                continue
            else:
                # applying of upsampling trainng set
                X_US = np.zeros((numSam_max - np.where(y_loc == i)[0].shape[0], X_loc.shape[1],))
                X_minor = X_loc[np.where(y_loc == i)[0]]
                y_minor = np.zeros((X_US.shape[0]))
                y_minor[:] = i

                for j in range(X_US.shape[0]):
                    ind = np.random.randint(0, X_minor.shape[0])
                    X_US[j, :] = X_minor[ind, :]

                X_loc = np.concatenate((X_loc, X_US))
                y_loc = np.concatenate((y_loc, y_minor))

        # random permutation
        temp = np.zeros((X_loc.shape[0], X_loc.shape[1] + 1))
        temp[:, 0 : X_loc.shape[1]] = X_loc
        temp[:, X_loc.shape[1]] = y_loc[:]

        temp = np.take(temp, np.random.permutation(temp.shape[0]), axis=0, out=temp)

        X_loc = temp[:, 0 : X_loc.shape[1]]
        y_loc[:] = temp[:, X_loc.shape[1]]

        return X_loc, y_loc

    def cat_permutations(self, df, cat_features, n_perm=2):

        inter_list = []
        for i in range(2, n_perm + 1):
            inter_list = inter_list + list(combinations(cat_features, i))

        for j in range(len(inter_list)):
            c = ''
            val = ''
            for i in inter_list[j]:
                c = c + ' ' + i
                val = val + ' ' + df[i].astype(str)
            df[c] = val
            cat_features = cat_features + [c]

        return df, cat_features


# TODO: need to refactor to upload large datasets in batches
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
