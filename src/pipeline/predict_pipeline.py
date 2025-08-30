import sys
import os
import pandas as pd
import numpy as np

# Adjusting the path to import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    This class is responsible for making predictions using the trained model.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Loads the model and preprocessor to make a prediction on the input features.
        The prediction is inverse-transformed (np.exp) to get the actual price.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            data_scaled = preprocessor.transform(features)
            log_preds = model.predict(data_scaled)
            
            # Inverse transform the log prediction to get the actual sale price
            preds = np.exp(log_preds)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    This class is responsible for mapping input data (e.g., from a web form)
    to the feature names that the model expects.
    """
    def __init__(self, **kwargs):
        # Initialize all features with None or the provided value
        features_list = [
            'MS SubClass', 'MS Zoning', 'Lot Frontage', 'Lot Area', 'Street', 'Lot Shape', 'Land Contour', 
            'Utilities', 'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2', 
            'Bldg Type', 'House Style', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 
            'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Mas Vnr Area', 
            'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 
            'BsmtFin Type 1', 'BsmtFin SF 1', 'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 
            'Total Bsmt SF', 'Heating', 'Heating QC', 'Central Air', 'Electrical', '1st Flr SF', 
            '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 
            'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual', 
            'TotRms AbvGrd', 'Functional', 'Fireplaces', 'Fireplace Qu', 'Garage Type', 
            'Garage Yr Blt', 'Garage Finish', 'Garage Cars', 'Garage Area', 'Garage Qual', 
            'Garage Cond', 'Paved Drive', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', 
            '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold', 
            'Sale Type', 'Sale Condition'
        ]
        for feature in features_list:
            # Replace spaces and slashes with underscores for valid attribute names
            attr_name = feature.replace(' ', '_').replace('/', '_')
            setattr(self, attr_name, kwargs.get(feature))

    def get_data_as_data_frame(self):
        """
        Converts the custom data object into a pandas DataFrame.
        """
        try:
            custom_data_input_dict = {
                'MS SubClass': [getattr(self, 'MS_SubClass', None)],
                'MS Zoning': [getattr(self, 'MS_Zoning', None)],
                'Lot Frontage': [getattr(self, 'Lot_Frontage', None)],
                'Lot Area': [getattr(self, 'Lot_Area', None)],
                'Street': [getattr(self, 'Street', None)],
                'Lot Shape': [getattr(self, 'Lot_Shape', None)],
                'Land Contour': [getattr(self, 'Land_Contour', None)],
                'Utilities': [getattr(self, 'Utilities', None)],
                'Lot Config': [getattr(self, 'Lot_Config', None)],
                'Land Slope': [getattr(self, 'Land_Slope', None)],
                'Neighborhood': [getattr(self, 'Neighborhood', None)],
                'Condition 1': [getattr(self, 'Condition_1', None)],
                'Condition 2': [getattr(self, 'Condition_2', None)],
                'Bldg Type': [getattr(self, 'Bldg_Type', None)],
                'House Style': [getattr(self, 'House_Style', None)],
                'Overall Qual': [getattr(self, 'Overall_Qual', None)],
                'Overall Cond': [getattr(self, 'Overall_Cond', None)],
                'Year Built': [getattr(self, 'Year_Built', None)],
                'Year Remod/Add': [getattr(self, 'Year_Remod_Add', None)],
                'Roof Style': [getattr(self, 'Roof_Style', None)],
                'Roof Matl': [getattr(self, 'Roof_Matl', None)],
                'Exterior 1st': [getattr(self, 'Exterior_1st', None)],
                'Exterior 2nd': [getattr(self, 'Exterior_2nd', None)],
                'Mas Vnr Type': [getattr(self, 'Mas_Vnr_Type', None)],
                'Mas Vnr Area': [getattr(self, 'Mas_Vnr_Area', None)],
                'Exter Qual': [getattr(self, 'Exter_Qual', None)],
                'Exter Cond': [getattr(self, 'Exter_Cond', None)],
                'Foundation': [getattr(self, 'Foundation', None)],
                'Bsmt Qual': [getattr(self, 'Bsmt_Qual', None)],
                'Bsmt Cond': [getattr(self, 'Bsmt_Cond', None)],
                'Bsmt Exposure': [getattr(self, 'Bsmt_Exposure', None)],
                'BsmtFin Type 1': [getattr(self, 'BsmtFin_Type_1', None)],
                'BsmtFin SF 1': [getattr(self, 'BsmtFin_SF_1', None)],
                'BsmtFin Type 2': [getattr(self, 'BsmtFin_Type_2', None)],
                'BsmtFin SF 2': [getattr(self, 'BsmtFin_SF_2', None)],
                'Bsmt Unf SF': [getattr(self, 'Bsmt_Unf_SF', None)],
                'Total Bsmt SF': [getattr(self, 'Total_Bsmt_SF', None)],
                'Heating': [getattr(self, 'Heating', None)],
                'Heating QC': [getattr(self, 'Heating_QC', None)],
                'Central Air': [getattr(self, 'Central_Air', None)],
                'Electrical': [getattr(self, 'Electrical', None)],
                '1st Flr SF': [getattr(self, '1st_Flr_SF', None)],
                '2nd Flr SF': [getattr(self, '2nd_Flr_SF', None)],
                'Low Qual Fin SF': [getattr(self, 'Low_Qual_Fin_SF', None)],
                'Gr Liv Area': [getattr(self, 'Gr_Liv_Area', None)],
                'Bsmt Full Bath': [getattr(self, 'Bsmt_Full_Bath', None)],
                'Bsmt Half Bath': [getattr(self, 'Bsmt_Half_Bath', None)],
                'Full Bath': [getattr(self, 'Full_Bath', None)],
                'Half Bath': [getattr(self, 'Half_Bath', None)],
                'Bedroom AbvGr': [getattr(self, 'Bedroom_AbvGr', None)],
                'Kitchen AbvGr': [getattr(self, 'Kitchen_AbvGr', None)],
                'Kitchen Qual': [getattr(self, 'Kitchen_Qual', None)],
                'TotRms AbvGrd': [getattr(self, 'TotRms_AbvGrd', None)],
                'Functional': [getattr(self, 'Functional', None)],
                'Fireplaces': [getattr(self, 'Fireplaces', None)],
                'Fireplace Qu': [getattr(self, 'Fireplace_Qu', None)],
                'Garage Type': [getattr(self, 'Garage_Type', None)],
                'Garage Yr Blt': [getattr(self, 'Garage_Yr_Blt', None)],
                'Garage Finish': [getattr(self, 'Garage_Finish', None)],
                'Garage Cars': [getattr(self, 'Garage_Cars', None)],
                'Garage Area': [getattr(self, 'Garage_Area', None)],
                'Garage Qual': [getattr(self, 'Garage_Qual', None)],
                'Garage Cond': [getattr(self, 'Garage_Cond', None)],
                'Paved Drive': [getattr(self, 'Paved_Drive', None)],
                'Wood Deck SF': [getattr(self, 'Wood_Deck_SF', None)],
                'Open Porch SF': [getattr(self, 'Open_Porch_SF', None)],
                'Enclosed Porch': [getattr(self, 'Enclosed_Porch', None)],
                '3Ssn Porch': [getattr(self, '3Ssn_Porch', None)],
                'Screen Porch': [getattr(self, 'Screen_Porch', None)],
                'Pool Area': [getattr(self, 'Pool_Area', None)],
                'Misc Val': [getattr(self, 'Misc_Val', None)],
                'Mo Sold': [getattr(self, 'Mo_Sold', None)],
                'Yr Sold': [getattr(self, 'Yr_Sold', None)],
                'Sale Type': [getattr(self, 'Sale_Type', None)],
                'Sale Condition': [getattr(self, 'Sale_Condition', None)],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)