# README for Lab 2: REST API for the Iris model (Part 2)
## Overview  
This project extends the Iris classification REST API from Lab 1. In Lab 2, the following is implemented:
- Built a **Python client application** to test all API endpoints.
- Extended the API with a **batch test endpoint** (`/test`).
- **Dockerized** the entire application using a custom Dockerfile.
- Ran the app in a Docker container and tested it using the client.
- Pushed the Docker image to Docker Hub.
## Features Added in Lab 2

-  **Client-side testing:** Full Python client to call the API endpoints
-  **New endpoint:** `GET /iris/model/<n>/test?dataset=<m>` to batch test a trained model
-  **Docker deployment:** API and backend code run as a container
-  **Docker Hub publishing:** Image uploaded to Docker Hub for sharing and reuse
## Project Structure  
```
Lab2/
│── base_iris.py # Machine learning model implementation (extended with test())
│── base_iris_flask.py # Flask REST API
│── requirements.txt # python dependencies
│── Dockerfile # Docker image instructions
│── client_session.log # client console log output when running on docker
│── client_log.txt # client console log output when running locally
│── docker_session.log # python session run from inside the container 
│── docker_hub.png # Screenshot of pushed image on docker Hub
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
| **GET** | `/iris/model/<n>/test?dataset=<m>` | Run a batch test on model `n` with dataset `m`. |
### **4. Running the Client** 
Make sure the API is running (Docker container or locally), then run:
```
python client.py
```
### **5. Docker Deployment** 
1. Build the image: `docker build -t iris-api .`
2. Run the container: `docker run -p 4000:4000 --name iris-container iris-api`
3. Open interactive shell: `docker exec -it iris-container bash`