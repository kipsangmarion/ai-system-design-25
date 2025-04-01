import requests

BASE_URL = "http://localhost:4000"

def upload_dataset(file_path):
    files = {'train': open(file_path, 'rb')}
    response = requests.post(f"{BASE_URL}/iris/datasets", files=files)
    print("Upload response:", response.json())
    return response.json()["dataset_id"]

def create_model(dataset_id):
    response = requests.post(f"{BASE_URL}/iris/model", json={"dataset": dataset_id})
    print("Model creation response:", response.json())
    return response.json()["model_id"]

def retrain_model(model_id, dataset_id):
    response = requests.put(f"{BASE_URL}/iris/model/{model_id}?dataset={dataset_id}")
    print("Retrain response:", response.json())
    return response.json()

def score_sample(model_id, features):
    features_str = ",".join(map(str, features))
    response = requests.get(f"{BASE_URL}/iris/model/{model_id}/score?fields={features_str}")
    print("Score response:", response.json())
    return response.json()

def test_model(model_id, dataset_id):
    response = requests.get(f"{BASE_URL}/iris/model/{model_id}/test?dataset={dataset_id}")
    print("Test response:", response.json())
    return response.json()

def main():
    print("=== Iris Model Client ===")

    # 1. Upload dataset
    dataset_id = upload_dataset("iris_extended_encoded.csv")

    # 2. Create model
    model_id = create_model(dataset_id)

    # 3. Retrain model
    retrain_model(model_id, dataset_id)

    # 4. Score a sample input (20 example values)
    sample = [5.1, 3.5, 1.4, 0.2, 2.1, 3.3, 4.2, 5.0, 2.5, 3.1,
              1.1, 4.5, 3.2, 2.8, 5.3, 4.1, 3.6, 2.2, 1.9, 0.8]
    score_sample(model_id, sample)

    # 5. Test model (once you've implemented the test endpoint)
    try:
        test_model(model_id, dataset_id)
    except Exception as e:
        print("Test endpoint not yet implemented:", e)

if __name__ == "__main__":
    main()
