import pandas as pd
import numpy as np
import json
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import KNNImputer
from apps.core.logger import Logger


class Preprocessor:
    """
    *****************************************************************************
    *
    * filename:       Preprocessor.py
    * version:        1.0
    * author:
    * creation date:
    *
    *
    *
    * description:    Class to pre-process training and predict dataset
    *
    ****************************************************************************
    """

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'Preprocessor', mode)

    def get_data(self):
        """
        * method: get_data
        * description: method to read datafile
        * return: A pandas DataFrame
        *
        *
        * Parameters
        *   none:
        """
        try:
            # reading the data file
            self.logger.info('Start of reading dataset...')
            self.data = pd.read_csv(self.data_path + '_validation/InputFile.csv')
            self.logger.info('End of reading dataset...')
            return self.data
        except Exception as e:
            self.logger.exception('Exception raised while reading dataset: %s' + str(e))
            raise Exception()

    def save_encoded_data(self):
        """
            * method: get_data
            * description: method to save datafile
            * return: A pandas DataFrame
            *
            *
            * Parameters
            *   none:
            """
        try:
            # reading the data file
            self.logger.info('Start of saving dataset...')
            self.data.to_csv(self.data_path + '_encode/encoded.csv')
            self.logger.info('End of saving dataset...')
            return self.data
        except Exception as e:
            self.logger.exception('Exception raised while reading dataset: %s' + str(e))
            raise Exception()

    def drop_columns(self, data, columns):
        """
        * method: drop_columns
        * description: method to drop columns
        * return: A pandas DataFrame after removing the specified columns.
        *
        *
        * Parameters
        *   data:
        *   columns:
        """
        self.data = data
        self.columns = columns
        try:
            self.logger.info('Start of Droping Columns...')
            self.useful_data = self.data.drop(labels=self.columns, axis=1)  # drop the labels specified in the columns
            self.logger.info('End of Droping Columns...')
            return self.useful_data
        except Exception as e:
            self.logger.exception('Exception raised while Droping Columns:' + str(e))
            raise Exception()

    def replace_invalid_values_with_null(self, data):

        """
          Method Name: is_null_present
          Description: This method replaces invalid values i.e. '?' with null, as discussed in EDA.

                             """
        # self.data = data
        try:
            self.logger.info('Start of replacing invalid values...')
            for column in data.columns:
                count = data[column][data[column] == '?'].count()
                if count != 0:
                    data[column] = data[column].replace('?', np.nan)
            self.logger.info('end of replacing invalid values...')
            return data
        except Exception as e:
            self.logger.exception('Exception raised while replacing invalid values' + str(e))
            raise Exception()

    def is_null_present(self, data):
        """
        * method: is_null_present
        * description: method to check null values
        * return: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
        *
        * Parameters
        *   data:
        """
        self.null_present = False
        try:
            self.logger.info('Start of finding missing values...')
            self.null_counts = data.isna().sum()  # check for the count of null values per column
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break

            if (self.null_present):  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv(
                    self.data_path + '_validation/' + 'null_values.csv')  # storing the null column information to file
            self.logger.info('End of finding missing values...')
            return self.null_present
        except Exception as e:
            self.logger.exception('Exception raised while finding missing values:' + str(e))
            raise Exception()

    def impute_missing_values(self, data):
        """
        * method: impute_missing_values
        * description: method to impute missing values
        * return: none
        *
        * Parameters
        *   data:
        """
        self.data = data
        try:
            self.logger.info('Start of imputing missing values...')
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            self.new_array = imputer.fit_transform(self.data)  # impute the missing values
            # convert the nd-array returned in the step above to a Data frame
            self.new_data = pd.DataFrame(data=self.new_array, columns=self.data.columns)
            self.logger.info('End of imputing missing values...')
            return self.new_data
        except Exception as e:
            self.logger.exception('Exception raised while imputing missing values:' + str(e))
            raise Exception()

    def feature_encoding(self, data):
        """
        * method: feature_encoding
        * description: method to convert categiorical to numerical
        * return: none
        *
        *
        * Parameters
        *   data:
        """
        try:
            self.logger.info('Start of feature encoding...')
            self.new_data = data.select_dtypes(include=['object']).copy()
            # Using the dummy encoding to encode the categorical columns to numerical ones
            for col in self.new_data.columns:
                self.new_data = pd.get_dummies(self.new_data, columns=[col], prefix=[col], drop_first=True)

            self.logger.info('End of feature encoding...')
            return self.new_data
        except Exception as e:
            self.logger.exception('Exception raised while feature encoding:' + str(e))
            raise Exception()

    def encode_categorical_values(self, data):
        """
        Method Name: encodeCategoricalValues Description: This method encodes all the categorical values in the
        training set. Output: A Dataframe which has all the categorical values encoded.
        On Failure: Raise Exception


                      """
        try:
            self.logger.info('Start of encode Categorical Values ...')
            # We can map the categorical values like below:
            data['Gender'] = data['Gender'].map({'a': 0, 'b': 1})

            # columns with two categorical data have same value 'f' and 't'.

            data['PriorDefault'] = data['PriorDefault'].map({'f': 0, 't': 1})
            data['Employed'] = data['Employed'].map({'f': 0, 't': 1})
            data['DriversLicense'] = data['DriversLicense'].map({'f': 0, 't': 1})

            self.logger.info('end of encode Categorical Values...')
            return data
        except Exception as e:
            self.logger.exception('Exception raised while splitting features and label:' + str(e))
            raise Exception()

    def feature_selection(self, data):
        """
        * method: get_data
        * description: method to feature selection of dataset
        * return: A pandas DataFrame
        *
        *
        * Parameters
        *   none:
        """
        self.data = data
        try:
            self.logger.info('Start feature selection of dataset...')

            X = self.data.iloc[:, :-18]
            y = self.data['Approved']
            ordered_rank_features = SelectKBest(score_func=chi2, k='all')
            ordered_feature = ordered_rank_features.fit(X, y)

            data_scores = pd.DataFrame(ordered_feature.scores_, columns=["Score"])
            data_columns = pd.DataFrame(X.columns)

            features_rank = pd.concat([data_columns, data_scores], axis=1)

            features_rank.columns = ['Features', 'Score']
            features_rank.nlargest(10, 'Score').to_csv(self.data_path + '_encode/features_rank.csv')

            data1 = self.data[['PriorDefault', 'YearsEmployed', 'CreditScore', 'Income', 'Approved']]
            data1.to_csv(self.data_path + '_encode/feature_selection.csv')
            self.logger.info('End feature selection of dataset...')
            return data1
        except Exception as e:
            self.logger.exception('Exception raised while feature selection of dataset: %s' + str(e))
            raise Exception()

    def feature_select(self, data):
        """
            * method: get_data
            * description: method to feature selection of dataset
            * return: A pandas DataFrame
            *
            *
            * Parameters
            *   none:
            """
        self.data = data
        try:
            self.logger.info('Start feature selection of dataset...')

            data1 = self.data[['PriorDefault', 'YearsEmployed', 'CreditScore', 'Income']]
            self.logger.info('End feature selection of dataset...')
            return data1
        except Exception as e:
            self.logger.exception('Exception raised while feature selection of dataset: %s' + str(e))
            raise Exception()

    def split_features_label(self, data, label_name):
        """
        * method: split_features_label
        * description: method to separate features and label
        * return: none
        *
        * Parameters
        *   data:
        *   label_name:
        """
        self.data = data
        try:
            self.logger.info('Start of splitting features and label ...')
            self.X = self.data.drop(labels=label_name,
                                    axis=1)  # drop the columns specified and separate the feature columns
            self.y = self.data[label_name]  # Filter the Label columns
            self.logger.info('End of splitting features and label ...')
            return self.X, self.y
        except Exception as e:
            self.logger.exception('Exception raised while splitting features and label:' + str(e))
            raise Exception()

    def final_predictset(self, data):
        """
        * method: final_predictset
        * description: method to build final predict set by adding additional encoded column with value as 0
        * return: column_names, Number of Columns
        *
        * Parameters
        *   none:
        """
        try:
            self.logger.info('Start of building final predictset...')
            with open('apps/database/columns.json', 'r') as f:
                data_columns = json.load(f)['data_columns']
                f.close()
            df = pd.DataFrame(data=None, columns=data_columns)
            df_new = pd.concat([df, data], ignore_index=True, sort=False)
            data_new = df_new.fillna(0)
            self.logger.info('End of building final predictset...')
            return data_new
        except ValueError:
            self.logger.exception('ValueError raised while building final predictset')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while building final predictset')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while building final predictset: %s' % e)
            raise e

    def preprocess_trainset(self):
        """
        * method: preprocess_trainset
        * description: method to pre-process training data
        * return: none
        *
        *
        * Parameters
        *   none:
        """
        try:
            self.logger.info('Start of Preprocessing...')
            # get data into pandas data frame
            data = self.get_data()
            # drop unwanted columns
            data = self.drop_columns(data, ['ZipCode'])
            # Replacing '?' with nan
            data = self.replace_invalid_values_with_null(data)
            # handle Categorical Values
            data = self.encode_categorical_values(data)
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, ['Married', 'BankCustomer', 'Citizen', 'EducationLevel', 'Ethnicity'])
            # check if missing values are present in the data set
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = self.impute_missing_values(data)  # missing value imputation
            # feature engineering
            data1 = self.feature_selection(data)
            # create separate features and labels
            self.X, self.y = self.split_features_label(data1, label_name='Approved')
            self.logger.info('End of Preprocessing...')
            return self.X, self.y
        except Exception:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            raise Exception

    def preprocess_predictset(self):
        """
        * method: preprocess_predictset
        * description: method to pre-process prediction data
        * return: none
        *
        *
        * Parameters
        *   none:
        """
        try:
            self.logger.info('Start of Preprocessing...')
            # get data into pandas data frame
            data = self.get_data()
            # drop unwanted columns
            data = self.drop_columns(data, ['ZipCode'])
            # Replacing '?' with nan
            data = self.replace_invalid_values_with_null(data)
            # handle Categorical Values
            data = self.encode_categorical_values(data)
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, ['Married', 'BankCustomer', 'Citizen', 'EducationLevel', 'Ethnicity'])
            # check if missing values are present in the data set
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = self.impute_missing_values(data)  # missing value imputation
            # feature engineering
            data1 = self.feature_select(data)
            data = self.final_predictset(data1)
            self.logger.info('End of Preprocessing...')
            return data
        except Exception:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            raise Exception

    def preprocess_predict(self, data):
        """
        * method: preprocess_predict
        * description: method to pre-process prediction data
        * return: none
        *
        *
        * Parameters
        *   none:
        """
        try:
            self.logger.info('Start of Preprocessing...')
            data = self.encode_categorical_values(data)
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, ['Married', 'BankCustomer', 'Citizen', 'EducationLevel', 'Ethnicity'])
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = self.impute_missing_values(data)  # missing value imputation

            data = self.final_predictset(data)
            self.logger.info('End of Preprocessing...')
            return data
        except Exception:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            raise Exception
