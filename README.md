# Bitcoin Price Prediction: Quantitative Analysis Matrix

**Author:** Abdelhafid Ibn Mhand  
**Module:** Intelligent Systems  

## Project Overview

This repository contains the final project for the Intelligent Systems module. The objective of this project is to develop a robust, machine-learning-driven application capable of predicting Bitcoin price trajectories based on historical market data and advanced technical indicators.

The project is divided into two main components:
1. **The Research Environment**: A Notebook detailing the data science pipeline (data cleaning, feature engineering, and model training).
2. **The Inference Engine**: A `Streamlit` dashboard deployed for real-time manual inference and model benchmarking.

## The Problematic

Financial markets, and cryptocurrency markets in particular, are highly volatile, non-stationary, and heavily influenced by non-linear relationships. Traditional statistical forecasting methods often fail to capture the complex, hidden patterns in high-frequency trading data.

The core problematic addressed in this project is: **How can we leverage ensemble machine learning algorithms to map historical technical vectors (price action, volume, momentum, and volatility) into reliable short-term price predictions, filtering market noise from actual structural signals?**

## Methodology & Pipeline Steps

The development of this intelligent system followed a rigorous data science lifecycle:

### 1. Data Collection & Ingestion
The foundational dataset, comprising historical OHLCV (Open, High, Low, Close, Volume) records for Bitcoin, was sourced from Kaggle. This raw ledger forms the baseline of the quantitative matrix, capturing the exact moments of institutional and retail liquidity exchanges.

### 2. Data Cleaning & Preprocessing
Real-world financial data is often noisy and incomplete. The preprocessing phase (detailed in the notebook) included:
* **Handling Missing Values**: Forward-filling and interpolating missing rows to maintain the temporal sequence of the time-series data.
* **Outlier Detection**: Filtering extreme anomalies in volume spikes that could skew the model weights, using standard deviation thresholds.
* **Data Normalization**: Applying `StandardScaler` to ensure all features (from thousands of dollars in price to decimal values in momentum oscillators) contribute proportionately to the loss function.

### 3. Feature Engineering
Raw price is insufficient for accurate prediction. The following technical vectors were engineered to provide the models with context regarding trend, momentum, and volatility:
* **Relative Strength Index (RSI)**: To measure overbought/oversold conditions.
* **Moving Average Convergence Divergence (MACD)**: To capture trend direction and momentum shifts.
* **Simple Moving Averages (SMA)**: To establish baseline support and resistance levels.
* **Bollinger Bands**: To measure market volatility and standard deviation from the mean.

### 4. Model Training & Selection
Multiple topologies were tested to find the optimal balance between bias and variance:
* **Gradient Boosting (Primary)**: Selected for its superior ability to adapt to high-frequency, non-linear data and minimize sequential errors.
* **Random Forest (Secondary)**: Utilized for its robustness against overfitting and its ability to handle high-dimensional feature spaces.
* **Linear Regression (Baseline)**: Used strictly as a baseline to measure the performance gains achieved by the complex ensemble models.

### 5. Deployment (Streamlit Dashboard)
The final models and scalers were serialized (`.pkl`) and integrated into a Python-based web application. The UI allows users to input custom market parameters and view the system's projection and structural analysis.

## Repository Structure

    basic-bitcoin-prediction/
    │
    ├── app.py                   # Main Streamlit application and dashboard UI
    ├── requirements.txt          # Required Python dependencies
    ├── README.md                 # Project documentation
    │
    ├── notebook/                 # Research and development environment
    │   └── bitcoin_prediction.ipynb      # Data cleaning, EDA, and model training
    │
    ├── models/                   # Serialized ML models & data scalers
    │   ├── gradient_boosting_model.pkl
    │   ├── random_forest_model.pkl
    │   ├── linear_regression_model.pkl
    │   └── feature_scaler.pkl
    │
    ├── data/                     # Datasets
    │   └── refined_btc_data.csv          # Cleaned historical ledger data
    │
    └── assets/                   # Static application assets
        └── bitcoin.png           

## Local Setup & Execution

### 1. Clone the repository
    git clone https://github.com/Hvllvix/basic-bitcoin-prediction.git
    cd basic-bitcoin-prediction

### 2. Install dependencies
It is highly recommended to run this project inside a virtual environment.
    
    pip install -r requirements.txt

### 3. Run the Application
Execute the following command to launch the Intelligent Systems dashboard locally:
    
    streamlit run app.py

### 4. Explore the Research Notebook
To view the data cleaning and training methodology:
    
    jupyter notebook notebook/bitcoin_prediction.ipynb

## Academic Disclaimer
This project is submitted in fulfillment of the Intelligent Systems module requirements. The predictive models contained within are designed for educational, research, and statistical analysis purposes. Cryptocurrencies are highly unpredictable, and this tool should not be interpreted as financial advice.
