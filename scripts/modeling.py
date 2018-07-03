import pandas as pd
import numpy as np
import os
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
import random
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop
from keras.layers.normalization import BatchNormalization
import keras

class modeling(object):

    def __init__(self):
        self.RANDOM_STATE = 1992

    def load_data(self):
        print('\tLoading preprocessed train dataset...')
        train = pd.read_csv('../data/ftsel-train.csv', encoding='utf-8', sep=',')
        self.len_train = len(train)
        # test = pd.read_csv('../data/ftsel-test.csv', encoding='utf-8', sep=',')
        print('\tTrain shape = {0}'.format(train.shape))
        print('\tLoaded.\n')

        return train

    def rmsle(self, y_true, y_pred):
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        return np.sqrt((1 / len(y_true)) * np.sum((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

    def stack_predict(self, models, X_test):
        # print('\tStacking predictions...\n')
        n_models = len(models)
        model_dict = dict()
        for i in range(n_models):
            model_dict['model_{}'.format(str(i))] = models[i].predict(X_test).tolist()
        all_preds = pd.DataFrame(model_dict)
        stack_preds = list()
        n = 0
        for j in all_preds:
            if n == 0:
                stack_preds = all_preds[j]
            else:
                stack_preds *= all_preds[j]
            n += 1
        return np.expm1((stack_preds) ** (1 / n_models))

    def single_predict(self, models, X_test):
        preds = models[0].predict(X_test).tolist()
        return np.expm1(preds)

    # def stack_predict_meta(self, models, meta_model, X_test, y_test):
    #     # print('\tStacking predictions...\n')
    #     n_models = len(models)
    #     model_dict = dict()
    #     for i in range(n_models):
    #         model_dict['model_{}'.format(str(i))] = models[i].predict(X_test).tolist()
    #     all_preds = pd.DataFrame(model_dict)
    #     # X_train2, X_test2, y_train2, y_test2 = train_test_split(X_test,
    #     #                                                         y_test,
    #     #                                                         test_size=0.3,
    #     #                                                         random_state=self.RANDOM_STATE)
    #     meta_model.fit(all_preds, y_test)
    #
    #     # stack_preds = list()
    #     # n = 0
    #     # for j in all_preds:
    #     #     if n == 0:
    #     #         stack_preds = all_preds[j]
    #     #     else:
    #     #         stack_preds *= all_preds[j]
    #     #     n += 1
    #     return np.expm1((stack_preds) ** (1 / n_models))

    def save_model(self, model, name, score):
        score = round(score, 4)
        print('\tSaving model {0} with RMSLE = {1}'.format(name, score))
        if not os.path.exists('../models/'):
            os.makedirs('../models/')
        joblib.dump(model, '../models/{0}_{1}.pkl'.format(score, name))

    def predictions(self, models, x_cols, CS=1e4):
        print('\tPredictions in Test set with Chunk Size = {}...\n'.format(str(CS)))
        preds_test = list()
        IDs = list()
        reader = pd.read_table('../data/ftsel-test.csv', sep=',', chunksize=CS)
        n = 0
        for chunk in reader:
            IDs += chunk['ID'].tolist()
            X_test = chunk.loc[:, x_cols]
            if len(models) == 1:
                preds_test += self.single_predict(models, X_test).tolist()
            else:
                preds_test += self.stack_predict(models, X_test).tolist()
            n += 1
        return preds_test, IDs

    def submit(self, name, preds_test, IDs, score):
        print('\tSubmissing predictions...')
        score = round(score, 4)
        if not os.path.exists('../submissions/'):
            os.makedirs('../submissions/')
        with open('../submissions/{0}_{1}.csv'.format(score, name), 'w') as f:
            n = 0
            f.write("ID,target\n")
            for _ in range(len(preds_test)):
                f.write("%s,%s\n" % (IDs[n], preds_test[n]))
                n += 1
        f.close()

    def improve_predictions(self, y, preds, alpha0 = 0.014, alphaf = 0.022, step = 0.0001):
        print('\tImproving predictions with alpha1 and alpha2...')
        best_alpha1 = 0
        best_alpha2 = 0
        rango = np.arange(alpha0, alphaf, step)
        best_score = self.rmsle(y, preds)
        for alpha in rango:
            preds2 = [i + (i * alpha) if i > 15.5 else i for i in preds]
            new_score = self.rmsle(y, preds2)
            if new_score < best_score:
                best_alpha1 = alpha
                best_score = new_score
        preds = [i + (i * best_alpha1) if i > 15.5 else i for i in preds]
        for alpha in rango:
            preds2 = [i - (i * alpha) if i < 13 else i for i in preds]
            new_score = self.rmsle(y, preds2)
            if new_score < best_score:
                best_alpha2 = alpha
                best_score = new_score
        return best_alpha1, best_alpha2

    def load_model(self, dir):
        model = joblib.load('{}.pkl'.format(dir))
        return model

    def avg_2_models(self, preds1, preds2, y_test):
        rmsle_dict = dict()
        for i in range(5, 100, 5):
            x = i/100
            y = 1 - x
            preds1_2 = [i * x for i in preds1]
            preds2_2 = [i * y for i in preds2]
            preds = [sum(x) for x in zip(preds1_2, preds2_2)]
            rmsle_dict[str(x)] = float(self.rmsle(y_test, preds))
        x_min_rmsle = min(rmsle_dict, key=rmsle_dict.get)
        x = eval(x_min_rmsle)
        y = 1 - x
        return x, y, rmsle_dict[str(x)]

    def avg_3_models(self, preds1, preds2, preds3, y_test):
        rmsle_list = list()
        for j in range(0, 105, 5):
            for i in range(0, 105, 5):
                x = i/100
                y = abs(j/100-x)
                z = (x+y)-1
                preds1_2 = [i * x for i in preds1]
                preds2_2 = [i * y for i in preds2]
                preds3_2 = [i * z for i in preds3]
                preds = [sum(x) for x in zip(preds1_2, preds2_2, preds3_2)]
                rmsle_list.append([[x, y, z], [float(self.rmsle(y_test, preds))]])
        scores_list = list()
        for i in rmsle_list:
            scores_list.append(i[1][0])
        index_min = scores_list.index(min(scores_list))
        xf = rmsle_list[index_min][0][0]
        yf = rmsle_list[index_min][0][1]
        zf = rmsle_list[index_min][0][2]
        return xf, yf, zf


if __name__ == '__main__':
    print('\nSTARTING MODELING\n')
    REFIT = False
    METASTACKING = True
    SUBMIT = False

    model = modeling()
    if not os.path.exists('../logs'):
        os.makedirs('../logs')
    fecha = str(datetime.now()).split(" ")[0]
    hora = str(datetime.now()).split(" ")[1].split(".")[0].split(":")
    if not os.path.exists('../logs/' + fecha):
        os.makedirs('../logs/' + fecha)
    logger = open("../logs/" + fecha + "/" + hora[0] + "-" + hora[1] + "-" + hora[2] + ".log", "w")
    logger.write('Best parameters found by cross-validation')
    logger.write('\n')

    train = model.load_data()
    x_cols = [x for x in train.columns if not x in ['ID', 'target']]
    X = train.loc[:, x_cols]
    y = train.loc[:, 'target']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=model.RANDOM_STATE)

    del train
    # del X
    # del y
    gc.collect()

    models = list()

    if REFIT:
        # GridSearching models parameters:#
        print('Training XGBoost model...')
        xgb_model = xgb.XGBRegressor()
        param_xgb = {'objective': ['reg:linear'],
                     "learning_rate": [0.05],
                     "max_depth": [7],
                     "n_estimators": [1000],
                     'subsample': [1],
                     "colsample_bytree": [0.55],
                     "colsample_bylevel": [0.9],
                     }
        xgboost_model = GridSearchCV(estimator=xgb_model,
                                     param_grid=param_xgb,
                                     n_jobs=-1,
                                     cv=3,
                                     verbose=5,
                                     scoring='neg_mean_absolute_error')
        xgboost_model.fit(X_train, y_train)
        preds_xgb = xgboost_model.best_estimator_.predict(X_test)
        score_xgb = model.rmsle(y_test, preds_xgb)
        print('Score XGB = {}\n'.format(score_xgb))
        logger.write('XGBoost:\n')
        logger.write('\tScore: {}\n'.format(str(score_xgb)))
        logger.write('\tBest params: {}\n'.format(str(xgboost_model.best_params_)))
        logger.write('\n')
        # Re-train with all cases
        xgboost_model.best_estimator_.fit(X, y)
        model.save_model(xgboost_model.best_estimator_, 'XGBoost', score_xgb)
        models.append(xgboost_model.best_estimator_)
        print('Submitting XGBoost single model')
        preds_test, IDs = model.predictions([xgboost_model.best_estimator_], x_cols, CS=1e5)
        model.submit('XGBoost', preds_test, IDs, score_xgb)

        print('Training RandomForest model...')
        rf_model = RandomForestRegressor(random_state=model.RANDOM_STATE)
        param_rf = {"max_depth": [2, 7, 10, 15],
                    "n_estimators": [1000],
                    # "min_samples_split": [0.55, 0.7, 0.9],
                    # "min_samples_leaf": [0.75, 0.9, 1],
                    }
        rf_mod = GridSearchCV(estimator=rf_model,
                                param_grid=param_rf,
                                n_jobs=-1,
                                cv=3,
                                verbose=5,
                              scoring='neg_mean_absolute_error')
        rf_mod.fit(X_train, y_train)
        preds_rf = rf_mod.best_estimator_.predict(X_test)
        score_rf = model.rmsle(y_test, preds_rf)
        print('Score RandomForest= {}\n'.format(score_rf))
        logger.write('Random Forest:\n')
        logger.write('\tScore: {}\n'.format(str(score_rf)))
        logger.write('\tBest params: {}\n'.format(str(rf_mod.best_params_)))
        logger.write('\n')
        # Re-train with all cases
        rf_mod.best_estimator_.fit(X, y)
        model.save_model(rf_mod.best_estimator_, 'RandomForest', score_rf)
        models.append(rf_mod.best_estimator_)
        print('Submitting RandomForest single model')
        preds_test, IDs = model.predictions([rf_mod.best_estimator_], x_cols, CS=1e5)
        model.submit('RandomForest', preds_test, IDs, score_rf)

        print('Training KNeighborsRegressor model...')
        knn_model = KNeighborsRegressor()
        param_knn = {"n_neighbors": [5, 7, 13],
                     "weights": ['uniform', 'distance'],
                       }
        knn_mod = GridSearchCV(estimator=knn_model,
                                 param_grid=param_knn,
                                 n_jobs=-1,
                                 cv=3,
                                 verbose=5,
                               scoring='neg_mean_absolute_error')
        knn_mod.fit(X_train, y_train)
        preds_knn = knn_mod.best_estimator_.predict(X_test)
        score_knn = model.rmsle(y_test, preds_knn)
        print('Score KNeighborsRegressor= {}\n'.format(score_knn))
        logger.write('KNeighborsRegressor:\n')
        logger.write('\tScore: {}\n'.format(str(score_knn)))
        logger.write('\tBest params: {}\n'.format(str(knn_mod.best_params_)))
        logger.write('\n')
        # Re-train with all cases
        knn_mod.best_estimator_.fit(X, y)
        model.save_model(knn_mod.best_estimator_, 'KNeighborsRegressor', score_knn)
        models.append(knn_mod.best_estimator_)
        print('Submitting KNeighborsRegressor single model')
        preds_test, IDs = model.predictions([knn_mod.best_estimator_], x_cols, CS=1e5)
        model.submit('KNeighborsRegressor', preds_test, IDs, score_knn)

        print('Training LightGBM model...')
        mdl_lgb = lgb.LGBMRegressor(n_jobs=1, verbosity = -1)
        param_lgbm = {'objective': ['regression'],
                      "boosting": ['gbdt'],
                      "metric": ['rmse'],
                      "learning_rate": [0.1, 0.01, 0.05],
                      'num_leaves': [15, 30, 100],
                      "feature_fraction": [0.55, 0.7, 0.75],
                      "bagging_fraction": [0.5, 0.6],
                      "max_depth": [3, 5, 7, 13],
                      'reg_alpha' : [1,1.2],
                      'reg_lambda' : [1,1.2,1.4],
                     }
        lgb_model = GridSearchCV(estimator=mdl_lgb,
                                     param_grid=param_lgbm,
                                     n_jobs=-1,
                                     cv=3,
                                     verbose=5,
                                     scoring='neg_mean_absolute_error')
        lgb_model.fit(X_train, y_train)
        preds_lgb = lgb_model.best_estimator_.predict(X_test)
        score_lgb = model.rmsle(y_test, preds_lgb)
        print('Score LGBM = {}\n'.format(score_lgb))
        logger.write('LightGBM:\n')
        logger.write('\tScore: {}\n'.format(str(score_lgb)))
        logger.write('\tBest params: {}\n'.format(str(lgb_model.best_params_)))
        logger.write('\n')
        # Re-train with all cases
        lgb_model.best_estimator_.fit(X, y)
        model.save_model(lgb_model.best_estimator_, 'LightGBM', score_lgb)
        models.append(lgb_model.best_estimator_)
        print('Submitting LightGBM single model')
        preds_test, IDs = model.predictions([lgb_model.best_estimator_], x_cols, CS=1e5)
        model.submit('LightGBM', preds_test, IDs, score_lgb)

        print('Training Lasso model...')
        lasso_model = Lasso(max_iter=2000, tol=0.01, random_state=model.RANDOM_STATE)
        param_lasso = {"alpha": [0.01, 0.1, 1, 10]
                       }
        lasso_mod = GridSearchCV(estimator=lasso_model,
                                param_grid=param_lasso,
                                n_jobs=-1,
                                cv=3,
                                verbose=5,
                                 scoring='neg_mean_absolute_error')
        lasso_mod.fit(X_train, y_train)
        preds_lasso = lasso_mod.best_estimator_.predict(X_test)
        score_lasso = model.rmsle(y_test, preds_lasso)
        print('Score Lasso= {}\n'.format(score_lasso))
        logger.write('Lasso:\n')
        logger.write('\tScore: {}\n'.format(str(score_lasso)))
        logger.write('\tBest params: {}\n'.format(str(lasso_mod.best_params_)))
        logger.write('\n')
        # Re-train with all cases
        lasso_mod.best_estimator_.fit(X, y)
        model.save_model(lasso_mod.best_estimator_, 'Lasso', score_lasso)
        models.append(lasso_mod.best_estimator_)
        print('Submitting Lasso single model')
        preds_test, IDs = model.predictions([lasso_mod.best_estimator_], x_cols, CS=1e5)
        model.submit('Lasso', preds_test, IDs, score_lasso)

        print('Training Keras model...')
        model_keras = Sequential()
        BatchNormalization()
        model_keras.add(Dense(1, input_dim=X_train.shape[1], activation='relu'))
        BatchNormalization()
        Dropout(0.2)
        model_keras.add(Dense(1028, activation='relu'))
        BatchNormalization()
        Dropout(0.2)
        model_keras.add(Dense(512, activation='relu'))
        BatchNormalization()
        Dropout(0.2)
        model_keras.add(Dense(100, activation='relu'))
        BatchNormalization()
        Dropout(0.2)
        model_keras.add(Dense(1))
        # model_keras.compile(optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0), loss = 'mean_squared_error')
        model_keras.compile(optimizer='adam', loss='mean_squared_error')
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        model_keras.fit(X_train, y_train, validation_data=(X_test, y_test),
                  nb_epoch=500,
                  batch_size=1,
                  verbose=2,
                  callbacks=[earlyStopping])
        preds_keras = model_keras.predict(X_test)[:,0]
        print(preds_keras)
        score_keras = model.rmsle(y_test, preds_keras)
        print('Score Keras= {}\n'.format(score_keras))
        # logger.write('Keras:\n')
        # logger.write('\tScore: {}\n'.format(str(score_ridge)))
        # logger.write('\tBest params: {}\n'.format(str(ridge_mod.best_params_)))
        # logger.write('\n')
        # Re-train with all cases
        model_keras.fit(X, y)
        # model.save_model(model_keras, 'Keras', score_keras)
        # models.append(model_keras)
        print('Submitting Keras single model')
        preds_test, IDs = model.predictions([model_keras], x_cols, CS=1e5)
        model.submit('Keras', preds_test, IDs, score_keras)


        print('Training Ridge model...')
        ridge_model = Ridge(random_state=model.RANDOM_STATE)
        param_ridge = {"alpha": [0.01, 0.1, 1, 10]
                       }
        ridge_mod = GridSearchCV(estimator=ridge_model,
                                 param_grid=param_ridge,
                                 n_jobs=-1,
                                 cv=3,
                                 verbose=5,
                                 scoring='neg_mean_absolute_error')
        ridge_mod.fit(X_train, y_train)
        preds_ridge = ridge_mod.best_estimator_.predict(X_test)
        score_ridge= model.rmsle(y_test, preds_ridge)
        print('Score Ridge= {}\n'.format(score_ridge))
        logger.write('Ridge:\n')
        logger.write('\tScore: {}\n'.format(str(score_ridge)))
        logger.write('\tBest params: {}\n'.format(str(ridge_mod.best_params_)))
        logger.write('\n')
        # Re-train with all cases
        ridge_mod.best_estimator_.fit(X, y)
        model.save_model(ridge_mod.best_estimator_, 'Ridge', score_ridge)
        models.append(ridge_mod.best_estimator_)
        print('Submitting Ridge single model')
        preds_test, IDs = model.predictions([ridge_mod.best_estimator_], x_cols, CS=1e5)
        model.submit('Ridge', preds_test, IDs, score_ridge)

    else:
        # For load simple models:#
        xgb = model.load_model('../models/1.4008_XGBoost')
        models.append(xgb)
        rf = model.load_model('../models/1.4053_RandomForest')
        models.append(rf)
        lgbm = model.load_model('../models/1.4001_LightGBM')
        models.append(lgbm)
        knn = model.load_model('../models/1.7726_KNeighborsRegressor')
        models.append(knn)


    if METASTACKING:
        n_models = len(models)
        model_dict_train = dict()
        model_dict_test = dict()
        print('Creating predictions data-frames')
        for i in range(n_models):
            model_dict_train['model_{}'.format(str(i))] = models[i].predict(X_train).tolist()
            model_dict_test['model_{}'.format(str(i))] = models[i].predict(X_test).tolist()
        all_preds_train = pd.DataFrame(model_dict_train)
        all_preds_test = pd.DataFrame(model_dict_test)
        meta_lasso = Lasso(random_state=model.RANDOM_STATE)
        print('Fitting Meta-Lasso')
        meta_lasso.fit(all_preds_train, y_train)
        preds_meta_lasso = meta_lasso.predict(all_preds_test)
        score_meta_lasso = model.rmsle(y_test, preds_meta_lasso)
        print('Score Meta-Lasso= {}\n'.format(score_meta_lasso)) # 0.556649 ????
        print(preds_meta_lasso)



    if SUBMIT:
        # Stacking models:#
        sp = model.stack_predict(models, X_test)
        sp = np.log1p(sp)
        score_stack = model.rmsle(y_test, sp)
        preds_test, IDs = model.predictions(models, x_cols, CS=1e5)
        model.submit('Stacking_lgbm_rf_xgb', preds_test, IDs, score_stack)
        logger.write('Stacking:\n')
        logger.write('\tScore: {}\n'.format(str(score_stack)))
        logger.write('\n')

        # Avg RF, XGB, LGBM models:# Best public solution!!!!!
        preds1 = xgb.predict(X_test)
        preds2 = rf.predict(X_test)
        preds3 = lgbm.predict(X_test)
        x, y, z = model.avg_3_models(preds1, preds2, preds3, y_test)
        preds_test1, IDs = model.predictions([xgb], x_cols, CS=1e5)
        preds_test2, _ = model.predictions([rf], x_cols, CS=1e5)
        preds_test3, _ = model.predictions([lgbm], x_cols, CS=1e5)
        preds_test1_2 = [i * x for i in preds_test1]
        preds_test2_2 = [i * y for i in preds_test2]
        preds_test3_2 = [i * z for i in preds_test3]
        preds_test = [sum(x) for x in zip(preds_test1_2, preds_test2_2, preds_test3_2)]
        model.submit('Avg_xgb{0}_rf{1}_lgbm{2}'.format(x, y, z), preds_test, IDs, score_stack)

        # preds_test1, IDs = model.predictions([xgb], x_cols, CS=1e5)
        # preds_test2, _ = model.predictions([rf], x_cols, CS=1e5)
        # preds_test3, _ = model.predictions([lgbm], x_cols, CS=1e5)
        # preds_test1_2 = [i * 0.3 for i in preds_test1]
        # preds_test2_2 = [i * 0.3 for i in preds_test2]
        # preds_test3_2 = [i * 0.4 for i in preds_test3]
        # preds_test = [sum(x) for x in zip(preds_test1_2, preds_test2_2, preds_test3_2)]
        # model.submit('Avg_xgb03_rf03_lgbm04', preds_test, IDs, 0)


        # Improving submissions:#
        # best_alpha1, best_alpha2 = model.improve_predictions(y_test, sp, alpha0=0.014, alphaf=0.022, step=0.0001)
        # preds_test = np.log1p(preds_test)
        # preds_test2 = [i+i*best_alpha1 if i > 15.5 else i for i in preds_test]
        # preds_test3 = [i-i*best_alpha2 if i < 13 else i for i in preds_test2]
        # preds_test3 = np.expm1(preds_test3)
        # model.submit('5withimp', preds_test3, IDs, score_stack)


        # Avg 2 models:#
        # preds1 = xgb.predict(X_test)
        # preds2 = rf.predict(X_test)
        # x, y, score_avg2 = model.avg_2_models(preds1, preds2, y_test)
        # preds_test1, IDs = model.predictions([xgb], x_cols, CS=1e5)
        # preds_test2, _ = model.predictions([rf], x_cols, CS=1e5)
        # preds_test1_2 = [i * x for i in preds_test1]
        # preds_test2_2 = [i * y for i in preds_test2]
        # preds_test = [sum(x) for x in zip(preds_test1_2, preds_test2_2)]
        # model.submit('Avg_xgb_rf', preds_test, IDs, score_avg2)


    logger.close()
    print('MODELING FINISHED')


else:
    print('Loading modeling functions\n')