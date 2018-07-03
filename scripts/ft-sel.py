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
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

class ftsel(object):

    def __init__(self):
        self.RANDOM_STATE = 1992

    def load_data(self):
        print('\tLoading preprocessed train and test datasets...')
        train = pd.read_csv('../data/pre-train.csv', encoding='utf-8', sep=',')
        self.len_train = len(train)
        test = pd.read_csv('../data/pre-test.csv', encoding='utf-8', sep=',')
        print('\tLoaded.\n')

        return train, test

    def save_data(self, train, test, name_tr = 'ftsel-train', name_te = 'ftsel-test'):
        print('\tTrain shape = {0}, Test shape = {1}'.format(train.shape, test.shape))
        print('\tSaving pre-processed train and test datasets...')
        train.to_csv('../data/{}.csv'.format(name_tr), sep=',', index=False)
        test.to_csv('../data/{}.csv'.format(name_te), sep=',', index=False)
        print('\tSaved.\n')

    def rmsle(self, y_true, y_pred):
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        return np.sqrt((1 / len(y_true)) * np.sum((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

    def sel_rec(self,X, y):
        print('\tSelection of features using recursive feature elimination and cross-validation')
        cols = [col for col in X.columns]
        # estimator = RandomForestRegressor(random_state=self.RANDOM_STATE, n_jobs=-1)
        estimator = DecisionTreeRegressor(random_state=self.RANDOM_STATE)
        # selector = RFECV(estimator, step=1, cv=5, scoring='neg_mean_squared_error')
        selector = RFE(estimator, step=1, verbose=1)
        selector = selector.fit(X, y)
        print("Optimal number of features : %d" % selector.n_features_)
        sup = selector.support_
        selection = list()
        for i, j in zip(cols, sup):
            if j:
                selection.append(i)

        return selection

if __name__ == '__main__':
    print('\nSTARTING FEATURE SELECTION\n')
    sel = ftsel()

    print('Loading data...\n')
    train, test = sel.load_data()
    x_cols = [x for x in train.columns if not x in ['ID', 'target']]
    subsample = train.sample(frac=0.4, random_state=sel.RANDOM_STATE)
    X = subsample.loc[:, x_cols]
    y = subsample.loc[:, 'target']
    del subsample
    gc.collect()

    print('Selecting features...\n')
    best_features = sel.sel_rec(X, y)
    del X
    del y
    gc.collect()

    best_features = best_features + ['ID'] + ['target']
    print('Saving data\n')
    sel.save_data(train = train.loc[:, best_features], test = test.loc[:, best_features])

    print('FEATURE SELECTION FINISHED')


else:
    print('Loading ft selection functions\n')