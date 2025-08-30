import os
import sys

# This line is added to ensure that the script can find the 'src' directory
# and import modules from it, like 'exception', 'logger', and the components.
# It assumes you run the script from the root of your project directory.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


if __name__ == "__main__":
    """
    This is the main execution block that runs the entire training pipeline.
    """
    try:
        logging.info("Starting the training pipeline.")

        # Step 1: Data Ingestion
        # This will read the raw data, split it into train/test sets, and save them as CSV files.
        logging.info("Initiating Data Ingestion.")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train data at: {train_data_path}, Test data at: {test_data_path}")

        # Step 2: Data Transformation
        # This will apply preprocessing (imputing, scaling, encoding) to the train and test data.
        logging.info("Initiating Data Transformation.")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data Transformation completed.")

        # Step 3: Model Training
        # This will train multiple models, select the best one based on R2 score, and save it.
        logging.info("Initiating Model Training.")
        model_trainer = ModelTrainer()
        r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model Training completed.")

        print(f"Pipeline finished successfully. The R2 score of the best model is: {r2_square}")

    except Exception as e:
        logging.error(f"An error occurred in the training pipeline: {e}")
        raise CustomException(e, sys)

