import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Adjusting the path to import from the src directory
# This assumes your script is run from the root directory of the project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    Specifies the path to save the preprocessing object.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    """
    This class is responsible for the data transformation process.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates and returns a data transformation pipeline (preprocessor object).
        
        The pipeline will:
        1. Impute missing values (mean for numerical, mode for categorical).
        2. Scale numerical features.
        3. One-hot encode categorical features.
        """
        try:
            # These are the columns identified from your EDA notebook for modeling
            numerical_columns = [
                'MS SubClass', 'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 
                'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 
                'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 
                'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath', 
                'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 
                'Garage Cars', 'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', 
                '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold'
            ]
            
            categorical_columns = [
                'MS Zoning', 'Street', 'Lot Shape', 'Land Contour', 'Utilities', 'Lot Config', 
                'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 
                'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 
                'Mas Vnr Type', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond', 
                'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating', 'Heating QC', 
                'Central Air', 'Electrical', 'Kitchen Qual', 'Functional', 'Fireplace Qu', 
                'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Paved Drive', 
                'Sale Type', 'Sale Condition'
            ]

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Pipeline for numerical features: Impute with mean, then scale.
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical features: Impute with mode, one-hot encode, then scale.
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # ColumnTransformer to apply different pipelines to different columns
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This method initiates the data transformation process.
        
        It reads train and test data, applies the preprocessing pipeline,
        and saves the preprocessor object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "SalePrice"

            # Separate features (X) and target (y) for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate features (X) and target (y) for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Apply the transformer to the datasets
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Log transform the target variable to handle skewness, as done in the notebook
            log_target_train = np.log(target_feature_train_df)
            log_target_test = np.log(target_feature_test_df)

            # Combine transformed features and target variable into a single array
            # Convert to dense array if the result is sparse, otherwise use asarray
            import scipy.sparse
            if scipy.sparse.issparse(input_feature_train_arr):
                train_features = input_feature_train_arr.toarray() # type: ignore
            else:
                train_features = np.asarray(input_feature_train_arr)
            if scipy.sparse.issparse(input_feature_test_arr):
                test_features = input_feature_test_arr.toarray() # type: ignore
            else:
                test_features = np.asarray(input_feature_test_arr)

            train_arr = np.c_[train_features, np.array(log_target_train)]
            test_arr = np.c_[test_features, np.array(log_target_test)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

