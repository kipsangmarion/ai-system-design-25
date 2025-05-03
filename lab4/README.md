# README for Lab 4: Model Retraining via AWS DynamoDB Streams (Part 4)
## Overview  
This lab demonstrates an end-to-end pipeline that connects a Dash frontend with a Flask API, TensorFlow model, and AWS backend services. It uses **DynamoDB streams** and **Lambda functions** to support dynamic model retraining when prediction errors occur or prediction confidence is low.
## Project Structure  
```
Lab3/
├── app_lab3_template.py     # Main Dash application
│── base_iris.py # Machine learning model implementation
│── base_iris_flask.py # Flask REST API
│── lab4_header.py # AWS configurations file
│── lambda.py # lambda function file
│── screenshots/
│──── cloud_logs.png # Cloud logs from AWS
│──── iris_extended_retrain.png # DynamoDB table with misclassified points and points with low confidence
│──── iris_extended_score.png # DynamoDB table with model predictions for all points
│──── lambda_trigger.png # Lambda function trigger
│── README.md # Project documentation (this file)
```
## Components

### Model
- A multi-layer perceptron (MLP) model is implemented using TensorFlow/Keras.
- Trained on the `iris_extended_encoded.csv` dataset.
- Model logic is defined in `base_iris.py`.

### Dash Frontend
- Allows users to:
  - Upload training data
  - Build/train/retrain models
  - Score one row at a time
  - Evaluate test datasets
- Implements loading indicators and plots (e.g., histogram, scatter, confusion matrix).
- Communicates with the Flask backend via HTTP.

### Flask API
- Exposes endpoints to:
  - Upload datasets
  - Create/retrain models
  - Score records
  - Run tests
- Defined in `base_iris_flask.py`.

### AWS Backend
- **DynamoDB Tables**
  - `IrisExtendedScore`: logs every prediction (row of features, class, actual, probability).
  - `IrisExtendedRetrain`: stores misclassified or low-confidence records.
- **Lambda Function**
  - Triggered by insert events in `IrisExtendedScore` (via DynamoDB Stream).
  - Filters incorrect or low-confidence entries and inserts them into `IrisExtendedRetrain`.

## Flow Diagram
```
Dash UI --> Flask API --> base_iris.py | v AWS DynamoDB (Score Table) | (Stream & Lambda Trigger) | v AWS DynamoDB (Retrain Table)
```
## How to Run Locally

1. Start Flask server:
   ```bash
   python base_iris_flask.py
2. Start Dash frontend:
   ```bash
   python app_lab3_template.py
3. Open the Dash UI at http://localhost:8050.