import pandas as pd
import numpy as np
#import time
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScaler
import gc
#from sklearn import model_selection
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.manifold import TSNE
#from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
#from sklearn.model_selection import KFold
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
#from sklearn.preprocessing import normalize
#from sklearn.manifold import TSNE
#import lightgbm as lgb
#import xgboost as xgb

class preprocessing(object):

    def __init__(self):
        self.RANDOM_STATE = 1992

    def load_data(self):
        print('\tLoading train and test datasets...')
        train = pd.read_csv('../data/train.csv', encoding='utf-8', sep=',')
        self.len_train = len(train)
        test = pd.read_csv('../data/test.csv', encoding='utf-8', sep=',')
        all_df = pd.concat((train, test), axis=0)
        all_df['target'] = np.log1p(all_df['target'])
        print('\tLoaded.\n')

        return all_df

    def save_data(self, a_df, name_tr = 'pre-train', name_te = 'pre-test'):
        print('\tAll Dataframe shape = {}'.format(a_df.shape))
        train_df = a_df.iloc[:self.len_train]
        test_df = a_df.iloc[self.len_train:]
        print('\tTrain shape = {0}, Test shape = {1}'.format(train_df.shape, test_df.shape))
        print('\tSaving pre-processed train and test datasets...')
        train_df.to_csv('../data/{}.csv'.format(name_tr), sep=',', index=False)
        test_df.to_csv('../data/{}.csv'.format(name_te), sep=',', index=False)
        print('\tSaved.\n')

    def drop_sparse(self, a_d):
        cols = [x for x in a_d.columns if not x in ['ID', 'target']]
        for col in cols:
            if len(np.unique(a_d[col])) < 2:
                a_d.drop(col, axis=1, inplace=True)
        return a_d

    def decomposition(self, a_d):
        N_COMP = 49

        print("\tStart decomposition process...")
        print("\t\tPCA")
        pca = PCA(n_components=N_COMP, random_state=self.RANDOM_STATE)
        pca_results_train = pca.fit_transform(a_d)

        print("\t\ttSVD")
        tsvd = TruncatedSVD(n_components=N_COMP, random_state=self.RANDOM_STATE)
        tsvd_results_train = tsvd.fit_transform(a_d)

        print("\t\tICA")
        ica = FastICA(n_components=N_COMP, random_state=self.RANDOM_STATE)
        ica_results_train = ica.fit_transform(a_d)

        print("\t\tGRP")
        grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=self.RANDOM_STATE)
        grp_results_train = grp.fit_transform(a_d)

        print("\t\tSRP")
        srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=self.RANDOM_STATE)
        srp_results_train = srp.fit_transform(a_d)

        print("\tAppend decomposition components to datasets...")
        for i in range(1, N_COMP + 1):
            a_d['pca_' + str(i)] = pca_results_train[:, i - 1]

            a_d['ica_' + str(i)] = ica_results_train[:, i - 1]

            a_d['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]

            a_d['grp_' + str(i)] = grp_results_train[:, i - 1]

            a_d['srp_' + str(i)] = srp_results_train[:, i - 1]
        print("\tDecomposition finished")
        return a_d


if __name__ == '__main__':
    print('\nSTARTING PREPROCESSING\n')
    pre = preprocessing()

    print('Loading data...\n')
    all_df = pre.load_data()

    X = all_df.drop(["ID", "target"], axis=1)
    y = all_df["target"]
    IDs = all_df["ID"]
    del all_df
    gc.collect()

    print('Drop sparse data...\n')
    X_drop = pre.drop_sparse(X)
    del X
    gc.collect()

    print('Decomposition\n')
    X_dec = pre.decomposition(X_drop)
    del X_drop
    gc.collect()

    print('Saving data...')
    pre.save_data(a_df = pd.concat([X_dec, y, IDs], axis=1))
    print('PREPROCESSING FINISHED')


else:
    print('Loading pre-processing functions\n')
