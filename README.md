# README for Lab 1: REST API for the Iris model
## Overview  
This project implements a RESTful API for training and interacting with an Iris classification model using Flask. The model is built using TensorFlow and trained on an extended Iris dataset with 20 features. The API allows users to upload datasets, create and train models, retrain existing models, and score new data samples.  
## Project Structure  
```
Lab1/
│── base_iris.py # Machine learning model implementation
│── base_iris_flask.py # Flask REST API
│── logs.txt # Log file capturing inputs and outputs
│── summary.txt # Summary of websites
│── screenshots/ # Folder containing Postman test screenshots
│── README.md # Project documentation (this file)
```
## Features  
- **Dataset Management:** Upload CSV training datasets.  
- **Model Training:** Create and train an Iris classification model.  
- **Model Retraining:** Re-train an existing model with a dataset.  
- **Scoring Predictions:** Use a trained model to classify new samples.  
- **Logging:** Captures all API interactions and debugging logs.  

## Setup Instructions  
### **1️. Install Dependencies**  
Ensure Python is installed, then install required packages:  
```bash
pip install flask tensorflow pandas numpy scikit-learn
```
### **2. Run the Flask API**  
Start the flask server:
```
python base_iris_flask.py
```
The website can be found here:
```
http://localhost:4000
```
### **3. API Endpoints** 
| **Method** | **Endpoint** | **Description** |
|------------|-------------|----------------|
| **POST** | `/iris/datasets` | Uploads a training dataset (CSV). |
| **POST** | `/iris/model` | Creates and trains a new model instance. |
| **PUT** | `/iris/model/<n>?dataset=<m>` | Retrains model `n` using dataset `m`. |
| **GET** | `/iris/model/<n>/score?fields=f1,f2,...,f20` | Scores input data with a trained model. |
