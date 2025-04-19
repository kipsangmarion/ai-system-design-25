from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# model functions
from base_iris import add_dataset, get_dataset, build, train, new_model, score, test

@app.route('/')
def home():
    return "Iris model API is running"

# upload training data
@app.route('/iris/datasets', methods=['POST'])
def upload_dataset():
    if 'train' not in request.files:
        return jsonify({"error": "No training dataset provided"}), 400

    file = request.files['train']
    df = pd.read_csv(file)

    dataset_id = add_dataset(df)
    return jsonify({"dataset_id": dataset_id})

# build and train a new model
@app.route('/iris/model', methods=['POST'])
def create_model():
    data = request.json
    dataset_id = data.get("dataset")

    if dataset_id is None:
        return jsonify({"error": "Dataset ID is required"}), 400

    model_id = new_model(dataset_id)
    return jsonify({"model_id": model_id})

# retrain an existing model
@app.route('/iris/model/<int:model_id>', methods=['PUT'])
def retrain_model(model_id):
    dataset_id = request.args.get("dataset")

    if dataset_id is None:
        return jsonify({"error": "Dataset ID is required"}), 400

    history = train(model_id, int(dataset_id))
    return jsonify({"Training_history": history})

# score model with provided input features
@app.route('/iris/model/<int:model_id>/score', methods=['GET'])
def score_model(model_id):
    fields = request.args.get("fields")

    if fields is None:
        return jsonify({"error": "Feature values required"}), 400

    try:
        feature_list = [float(x) for x in fields.split(",")]
    except ValueError:
        return jsonify({"Error": "All features must be numeric"}), 400

    result = score(model_id, feature_list)
    return jsonify({"prediction": result})

# test the model
@app.route('/iris/model/<int:model_id>/test', methods=['GET'])
def test_model_endpoint(model_id):
    dataset_id = request.args.get("dataset")
    
    if dataset_id is None:
        return jsonify({"error": "Dataset ID is required"}), 400

    try:
        result = test(model_id, int(dataset_id))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)