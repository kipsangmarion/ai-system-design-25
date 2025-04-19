# README for Lab 3: Interactive Dashboard for the Iris model (Part 3)
## Overview  
This project builds upon Labs 1 and 2 to create a fully interactive Dash dashboard for exploring, training, scoring, and testing an Iris classification model. The dashboard communicates with a Flask-based API backend and allows users to:
- Load and upload training datasets
- Build and retrain machine learning models
- Score custom input rows
- Test model performance and view results as a confusion matrix
## Project Structure  
```
Lab3/
├── app_lab3_template.py     # Main Dash application
│── base_iris.py # Machine learning model implementation
│── base_iris_flask.py # Flask REST API
│── logs/
│──── dash_log.log # Captured Dash console output
│──── flask_log.log # Captured API console output
│── screenshots/
│──── build_and_train_tab.png # Training tab screenshot
│──── explore_data_tab.png # Explore data tab screenshot
│──── score_model_tab.png # Score model tab screenshot
│──── test_data_tab.png # Testing tab screenshot
│── README.md # Project documentation (this file)
```
## How to Run the App  
### **1️. Start the Flask API**  
In one terminal, run:  
```bash
python base_iris_flask.py
```
### **2. Start the Dash UI**  
In another terminal, run:
```
python app_lab3_template.py
```
The website can be found here: http://127.0.0.1:8050/
## Key Features
1. **Explore Tab**:
	a. Load dataset from file or upload via drag-and-drop
	b. View histograms, scatter plots, and full data table
2. **Training Tab**
	a. Create new model by selecting dataset ID
	b. Retrain model using existing model ID
	c. Live training metrics shown using line chart
3. **Score Tab**
	a. Enter a row of feature values and model ID
	b. Receive real-time prediction from backend
4. **Test Tab**
	a. Evaluate model on selected dataset
	b. Visualize results using a confusion matrix with annotated accuracy
## Docker Support
If running via Docker:
```
docker build -t iris-dashboard .
docker run -p 4000:4000 iris-dashboard
```