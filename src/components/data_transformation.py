import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.transformation_config = DataTransformationConfig()
        self.numerical_columns = ['writing_score', 'reading_score']
        self.categorical_columns = [
            'gender',
            'race_ethnicity',
            'parental_level_of_education',
            'lunch',
            'test_preparation_course',
        ]
        self.target_column_name = 'math_score'

    def get_data_transformer_object(self) -> ColumnTransformer:
        '''
        This function is responsible for data transformation.
        It returns a preprocessor handling numerical and categorical
        columns differently.
        '''
        try:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False)),
            ])

            logging.info(f'Numerical columns :{self.numerical_columns}.')

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False)),
            ])

            logging.info(f'Categorical columns :{self.categorical_columns}.')

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, self.numerical_columns),
                ('cat_pipeline', cat_pipeline, self.categorical_columns),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(
                columns=[self.target_column_name], axis=1)
            target_feature_train_df = train_df[self.target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[self.target_column_name], axis=1)
            target_feature_test_df = test_df[self.target_column_name]

            logging.info(
                "Applying preprocessing object on training DataFrame and testing DataFrame."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(
                input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            save_object(
                file_path=self.transformation_config.preprocessing_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info("Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessing_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
