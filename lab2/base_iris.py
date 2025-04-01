import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

print('starting up iris model service')

global models, datasets
models = []
datasets = []
metrics = []

def build():
    global models

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
        ])

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    models.append( model )
    model_ID = len(models) - 1

    return model_ID

def load_local():
    global datasets

    print("load extended iris dataset")

    dataFolder = './'
    dataFile = dataFolder + "iris_extended_encoded.csv"

    datasets.append( pd.read_csv(dataFile) )
    return len( datasets ) - 1

def add_dataset( df ):
    global datasets

    datasets.append( df )
    return len( datasets ) - 1

def get_dataset( dataset_ID ):
    global datasets

    return datasets[dataset_ID]

def train(model_ID, dataset_ID):
    global datasets, models
    dataset = datasets[dataset_ID]
    model = models[model_ID]

    X = dataset.iloc[:,1:].apply(pd.to_numeric, errors='coerce').values
    y = dataset.iloc[:,0].values

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y = pd.get_dummies(y1).values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    history = model.fit(X_train, y_train, batch_size=1, epochs=10)
    print(history.history)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(X_test)

    actual = np.argmax(y_test,axis=1)
    predicted = np.argmax(y_pred,axis=1)
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")

    conf_matrix = confusion_matrix(actual, predicted)
    print('Confusion matrix on test data is {}'.format(conf_matrix))
    print('Precision Score on test data is {}'.format(precision_score(actual, predicted, average=None)))
    print('Recall Score on test data is {}'.format(recall_score(actual, predicted, average=None)))

    return(history.history)

# function to build and train a model dynamically with dataset ID
def new_model(dataset_ID):
    model_ID = build()
    train(model_ID, dataset_ID)
    return model_ID

def score( model_ID, feature_list ):
    global models
    model = models[model_ID]

    x_test2 = np.array( [feature_list] )

    y_pred2 = model.predict(x_test2)
    print(y_pred2)
    iris_class = np.argmax(y_pred2, axis=1)[0]
    print(iris_class)

    return "Score done, class=" + str(iris_class)

# functions for testing
def save_metrics(model_id, metrics_bundle):
    if model_id < len(metrics):
        metrics[model_id] = metrics_bundle
    else:
        metrics.append(metrics_bundle)
    return

def test(model_id, dataset_id):
    global models, datasets, metrics

    model = models[model_id]
    df = datasets[dataset_id]

    X_test = df.iloc[:, 1:21].values
    y = df.iloc[:, 0].values

    encoder = LabelEncoder()
    y1 = encoder.fit_transform(y)
    y_test = pd.get_dummies(y1).values

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(X_test)

    actual = np.argmax(y_test, axis=1)
    predicted = np.argmax(y_pred, axis=1)

    save_metrics(model_id, {
        'accuracy': float(accuracy),
        'actual': actual.tolist(),
        'predicted': predicted.tolist()
    })

    return metrics[model_id]