import sys
import os

# Add the project root directory to the Python path
# This is the definitive fix for the 'ModuleNotFoundError'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for the landing page
@app.route('/')
def index():
    # Renders your landing page
    return render_template('Index.html') 

## Route for the main prediction form
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # This displays the main prediction form when the user clicks "Get Started"
        return render_template('home.html')
    else:
        # This block executes when the user submits the prediction form
        data = CustomData(
            MS_Zoning=request.form.get('MS_Zoning'),
            Lot_Frontage=float(request.form.get('Lot_Frontage') or 0),
            Lot_Area=int(request.form.get('Lot_Area') or 0),
            Street=request.form.get('Street'),
            Alley=request.form.get('Alley'),
            Lot_Shape=request.form.get('Lot_Shape'),
            Land_Contour=request.form.get('Land_Contour'),
            Lot_Config=request.form.get('Lot_Config'),
            Land_Slope=request.form.get('Land_Slope'),
            Neighborhood=request.form.get('Neighborhood'),
            MS_SubClass=int(request.form.get('MS_SubClass') or 0),
            Overall_Qual=int(request.form.get('Overall_Qual') or 0),
            Overall_Cond=int(request.form.get('Overall_Cond') or 0),
            Year_Built=int(request.form.get('Year_Built') or 0),
            Year_RemodAdd=int(request.form.get('Year_RemodAdd') or 0),
            Bldg_Type=request.form.get('Bldg_Type'),
            House_Style=request.form.get('House_Style'),
            Foundation=request.form.get('Foundation'),
            Mas_Vnr_Type=request.form.get('Mas_Vnr_Type'),
            Mas_Vnr_Area=float(request.form.get('Mas_Vnr_Area') or 0),
            Bsmt_Qual=request.form.get('Bsmt_Qual'),
            Bsmt_Cond=request.form.get('Bsmt_Cond'),
            Bsmt_Exposure=request.form.get('Bsmt_Exposure'),
            BsmtFin_Type_1=request.form.get('BsmtFin_Type_1'),
            BsmtFin_SF_1=float(request.form.get('BsmtFin_SF_1') or 0),
            BsmtFin_Type_2=request.form.get('BsmtFin_Type_2'),
            BsmtFin_SF_2=float(request.form.get('BsmtFin_SF_2') or 0),
            Bsmt_Unf_SF=float(request.form.get('Bsmt_Unf_SF') or 0),
            Total_Bsmt_SF=float(request.form.get('Total_Bsmt_SF') or 0),
            First_Flr_SF=int(request.form.get('First_Flr_SF') or 0),
            Second_Flr_SF=int(request.form.get('Second_Flr_SF') or 0),
            Gr_Liv_Area=int(request.form.get('Gr_Liv_Area') or 0),
            Bsmt_Full_Bath=int(request.form.get('Bsmt_Full_Bath') or 0),
            Bsmt_Half_Bath=int(request.form.get('Bsmt_Half_Bath') or 0),
            Full_Bath=int(request.form.get('Full_Bath') or 0),
            Half_Bath=int(request.form.get('Half_Bath') or 0),
            Bedroom_AbvGr=int(request.form.get('Bedroom_AbvGr') or 0),
            Kitchen_AbvGr=int(request.form.get('Kitchen_AbvGr') or 0),
            Kitchen_Qual=request.form.get('Kitchen_Qual'),
            TotRms_AbvGrd=int(request.form.get('TotRms_AbvGrd') or 0),
            Fireplaces=int(request.form.get('Fireplaces') or 0),
            Fireplace_Qu=request.form.get('Fireplace_Qu'),
            Garage_Type=request.form.get('Garage_Type'),
            Garage_Yr_Blt=float(request.form.get('Garage_Yr_Blt') or 0),
            Garage_Finish=request.form.get('Garage_Finish'),
            Garage_Cars=int(request.form.get('Garage_Cars') or 0),
            Garage_Area=float(request.form.get('Garage_Area') or 0),
            Garage_Qual=request.form.get('Garage_Qual'),
            Garage_Cond=request.form.get('Garage_Cond'),
            Paved_Drive=request.form.get('Paved_Drive'),
            Wood_Deck_SF=int(request.form.get('Wood_Deck_SF') or 0),
            Open_Porch_SF=int(request.form.get('Open_Porch_SF') or 0),
            Enclosed_Porch=int(request.form.get('Enclosed_Porch') or 0),
            ThreeSsn_Porch=int(request.form.get('ThreeSsn_Porch') or 0),
            Screen_Porch=int(request.form.get('Screen_Porch') or 0),
            Pool_Area=int(request.form.get('Pool_Area') or 0),
            Pool_QC=request.form.get('Pool_QC'),
            Fence=request.form.get('Fence'),
            Misc_Feature=request.form.get('Misc_Feature'),
            Misc_Val=int(request.form.get('Misc_Val') or 0),
            Heating_QC=request.form.get('Heating_QC'),
            Central_Air=request.form.get('Central_Air'),
            Electrical=request.form.get('Electrical'),
            Mo_Sold=int(request.form.get('Mo_Sold') or 0),
            Yr_Sold=int(request.form.get('Yr_Sold') or 0),
            Sale_Type=request.form.get('Sale_Type'),
            Sale_Condition=request.form.get('Sale_Condition')
        )
        
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
