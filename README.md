# 🏡 Ames Housing Price Predictor

An intelligent web application that predicts housing prices in Ames, Iowa, using a machine learning model and a comprehensive set of property features.

🔗 **Live Demo:** [your-live-demo-link-here]

## 🌐 Web App Preview

- Landing Page
- Multi-Step Prediction Form
- Prediction Result

## 📁 Project Structure
```
Ames-Housing-Price-Prediction/
│
├── .github/workflows/
│   └── main.yaml                    # CI/CD pipeline for deployment
├── artifacts/                       # Saved model and preprocessor files
│   ├── data_transformation.pkl
│   └── model.pkl
├── notebook/                        # Jupyter notebooks for analysis and modeling
│   ├── 1. EDA.ipynb
│   └── 2. Model Training.ipynb
├── src/                             # Source code for the ML application
│   ├── components/                  # Core ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── __init__.py
│   ├── pipeline/                    # Prediction pipeline logic
│   │   ├── predict_pipeline.py
│   │   └── __init__.py
│   ├── exception.py                 # Custom exception handling
│   ├── logger.py                    # Logging configuration
│   ├── utils.py                     # Utility functions
│   └── __init__.py
├── templates/                       # HTML templates for the web interface
│   ├── Index.html
│   └── home.html
├── .gitignore                       # Git ignore file
├── app.py                           # Main Flask web application
├── README.md                        # Project documentation
├── requirements.txt                 # Project dependencies
└── setup.py                        # Project setup configuration
```


## 📝 Key Components

- **model.pkl:** The pre-trained regression model that predicts housing prices
- **data_transformation.pkl:** The preprocessor object for transforming user input
- **app.py:** The main Flask application that handles web routes and prediction logic
- **predict_pipeline.py:** A dedicated module to load the model and make predictions on new data
- **templates/:** Contains the HTML files for the user interface
- **notebook/:** Jupyter Notebooks detailing the exploratory data analysis (EDA) and model training process

## ⚙️ How It Works

1. **User Input:** The user navigates through a 7-step form on the web interface to provide detailed property features
2. **Data Processing:** The Flask backend receives the form data and uses a custom data pipeline to transform it into a format suitable for the model
3. **Prediction:** The processed data is fed into the trained machine learning model, which generates a price prediction
4. **Result Display:** The estimated house price is sent back to the web page and displayed to the user in a clean, readable format

## ✨ Features

- **Multi-Step Form:** An intuitive 7-step form for detailed and organized property feature input
- **Interactive UI:** A clean, responsive, and user-friendly interface built with Tailwind CSS
- **Real-Time Prediction:** Instant housing price estimates powered by a robust machine learning model
- **Data-Driven Accuracy:** The model is trained on the comprehensive Ames, Iowa housing dataset for reliable predictions

## 🔮 Future Updates

We are working to enhance the Ames Housing Price Predictor with the following upcoming features:

### 🗺️ Interactive Map Integration
Visualize property locations and compare prices across different neighborhoods in Ames.

### 📊 Advanced Analytics Dashboard
See how different features (e.g., square footage, number of bathrooms) impact the final price.

### 👤 User Accounts
Save and compare multiple property predictions.

## 🚦 Quick Start

Experience the Ames Housing Price Predictor in two ways:

### Try the Live Demo:
- Visit: [your-live-demo-link-here]
- No installation required

### Run Locally:

# Clone the repository
git clone https://github.com/Swayam-Burde/Ames-Housing-Price-Prediction.git
cd Ames-Housing-Price-Prediction

# Create and activate a virtual environment
# On Windows:
python -m venv venv
venv\Scripts\activate

# On macOS/Linux:
 python3 -m venv venv
 source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

Then, open your browser and go to http://127.0.0.1:5000.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request with any improvements or new features.

## ⚠️ Disclaimer

This tool is for informational purposes only and should not be used as a substitute for a professional real estate appraisal or financial advice. The predictions are based on historical data and may not reflect current market conditions.
