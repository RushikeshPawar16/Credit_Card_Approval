from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from apps.core.logger import Logger
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import r2_score


class ModelTuner:
    """
    *****************************************************************************
    *
    * filename:       model_tuner.py
    * version:        1.0
    * author:
    * creation date:
    *
    *
    * description:    Class to tune and select best model
    *
    ****************************************************************************
    """

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'ModelTuner', mode)
        self.rfc = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')
        # self.knn = KNeighborsClassifier()

    def best_params_randomforest(self, train_x, train_y):
        """
        * method: best_params_randomforest
        * description: method to get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
        * return: The model with the best parameters
        *
        *
        * Parameters
        *   train_x:
        *   train_y:
        """
        try:
            self.logger.info('Start of finding best params for randomforest algo...')
            # initializing with different combination of parameters
            self.param_grid = {
                "n_estimators": [2, 3, 4],
                "criterion": ['gini', 'entropy'],
                "max_depth": range(2, 4, 1),
                "max_features": ['auto', 'log2']}

            # self.cv_method = RepeatedStratifiedKFold(n_splits=5,
            #                                          n_repeats=3,
            #                                          random_state=3)
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.rfc, param_grid=self.param_grid, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.rfc = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.rfc.fit(train_x, train_y)
            self.logger.info('Random Forest best params: ' + str(self.grid.best_params_))
            self.logger.info('End of finding best params for randomforest algo...')

            return self.rfc
        except Exception as e:
            self.logger.exception('Exception raised while finding best params for randomforest algo:' + str(e))
            raise Exception()

    # def get_best_params_for_KNN(self, train_x, train_y):
    #     """
    #          Method Name: get_best_params_for_KNN
    #          Description: get the parameters for KNN Algorithm which give the best accuracy.
    #                                                          Use Hyper Parameter Tuning.
    #         Output: The model with the best parameters
    #         On Failure: Raise Exception
    #
    #
    #
    #                                     """
    #
    #     try:
    #         self.logger.info('Start of finding best params for KNN algo...')
    #         # initializing with different combination of parameters
    #         self.param_grid_knn = {
    #             'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    #             'leaf_size': [10, 17, 24, 28, 30, 35],
    #              'n_neighbors': [2, 3],
    #             'p': [1, 2]
    #         }
    #         # self.cv_method = RepeatedStratifiedKFold(n_splits=5,
    #         #                                          n_repeats=3,
    #         #                                          random_state=999)
    #         # Creating an object of the Grid Search class
    #         self.grid = GridSearchCV(self.knn, self.param_grid_knn, verbose=3,
    #                                  cv=5)
    #         # finding the best parameters
    #         self.grid.fit(train_x, train_y)
    #
    #         # extracting the best parameters
    #         self.algorithm = self.grid.best_params_['algorithm']
    #         self.leaf_size = self.grid.best_params_['leaf_size']
    #         self.n_neighbors = self.grid.best_params_['n_neighbors']
    #         self.p = self.grid.best_params_['p']
    #
    #         # creating a new model with the best parameters
    #         self.knn = KNeighborsClassifier(algorithm=self.algorithm, leaf_size=self.leaf_size,
    #                                         n_neighbors=self.n_neighbors, p=self.p, n_jobs=-1)
    #         # training the mew model
    #         self.knn.fit(train_x, train_y)
    #         self.logger.info('Knn Forest best params: ' + str(self.grid.best_params_))
    #         self.logger.info('End of finding best params for knn algo...')
    #
    #         return self.knn
    #     except Exception as e:
    #         self.logger.exception('Exception raised while finding best params for knn algo:' + str(e))
    #         raise Exception()

    def best_params_xgboost(self, train_x, train_y):
        """
        * method: best_params_xgboost
        * description: method to get the parameters for XGBoost Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
        * return: The model with the best parameters
        *
        * Parameters
        *   train_x:
        *   train_y:
        """
        try:
            self.logger.info('Start of finding best params for XGBoost algo...')
            # initializing with different combination of parameters
            self.param_grid_xgboost = {
                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), self.param_grid_xgboost, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(objective='binary:logistic', learning_rate=self.learning_rate,
                                     max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger.info('XGBoost best params: ' + str(self.grid.best_params_))
            self.logger.info('End of finding best params for XGBoost algo...')
            return self.xgb
        except Exception as e:
            self.logger.exception('Exception raised while finding best params for XGBoost algo:' + str(e))
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
        * method: get_best_model
        * description: method to get best model
        * return: none
        *
        *
        * Parameters
        *   train_x:
        *   train_y:
        *   test_x:
        *   test_y:
        """
        try:
            # self.logger.info('Start of finding best model...')
            # self.knn = self.get_best_params_for_KNN(train_x, train_y)
            # self.prediction_knn = self.knn.predict(test_x)  # Predictions using the Knn Model
            #
            # if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
            #     # will use accuracy in that case
            #     self.knn_score = accuracy_score(test_y, self.prediction_knn)
            #     self.logger.info('Accuracy for knn:' + str(self.knn_score))
            # else:
            #     self.knn_score = roc_auc_score(test_y, self.prediction_knn)  # AUC for knn
            #     self.logger.info('AUC for knn:' + str(self.knn_score))

            self.xgb = self.best_params_xgboost(train_x, train_y)
            self.prediction_xgb = self.xgb.predict(test_x)  # prediction using the xgb  Algorithm

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                self.xgb_score = accuracy_score(test_y, self.prediction_xgb)
                self.logger.info('Accuracy for Xgboost:' + str(self.xgb_score))
            else:
                self.xgb_score = roc_auc_score(test_y, self.prediction_xgb)  # AUC for XGBoost
                self.logger.info('AUC for Xgboost:' + str(self.xgb_score))

            # create best model for Random Forest

            self.random_forest = self.best_params_randomforest(train_x, train_y)
            self.prediction_random_forest = self.random_forest.predict(
                test_x)  # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We
                # will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y, self.prediction_random_forest)
                self.logger.info('Accuracy for Random Forest:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest)  # AUC for XGBoost
                self.logger.info('AUC for Random Forest:' + str(self.random_forest_score))

            # comparing the two models
            self.logger.info('End of finding best model...')
            if (self.random_forest_score < self.xgb_score):
                return 'XGB', self.xgb
            else:
                return 'RandomForest', self.random_forest

        except Exception as e:
            self.logger.exception('Exception raised while finding best model:' + str(e))
            raise Exception()
