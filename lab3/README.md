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

## Reflection Questions

### 1. What did you find easy to do in Dash?

Working with visual components like histograms, scatter plots, and tabular displays was intuitive using Dash. The `dcc.Graph`, `dash_table.DataTable`, and `dcc.Dropdown` components were easy to integrate and configure. Additionally, defining layout using Dash’s declarative structure made it straightforward to arrange the UI elements across multiple tabs.

---

### 2. Likewise, what was hard to implement or you didn’t wind up getting it to work?

The most challenging aspect was integrating real-time feedback for long-running operations like model training. Although Dash’s `dcc.Loading` simplified this, getting clear indicators during backend delays required careful planning. Handling state across tabs and managing file uploads in sync with API calls also involved debugging and precise callback structuring.

---

### 3. What other components, or what revised dashboard design, would you suggest to better assist in explaining the behavior of the Iris model to a client?

To improve the user experience, I would suggest including a confusion matrix summary with per-class precision and recall, as well as feature importance plots (e.g., bar charts) if available from the model. Adding a log pane that shows detailed API responses and model internals could help clients understand what’s happening behind the scenes. A dropdown to toggle between different models or training sessions would also make it more interactive for demonstrations.

---

### 4. Can you think of better ways to link the “back end” Iris model and its results with the front-end Dash functions?

One improvement could be persistent state management using `dcc.Store` or writing temporary logs to a database or cache. This would allow the backend to push status updates or metrics to the frontend without requiring manual refreshes. Additionally, switching to WebSockets or long-polling callbacks could offer more responsive updates during training or testing, especially for deployments beyond local development.
