# in case this is run outside of conda environment with python2
import mlflow
import argparse
import sys
from mlflow import pyfunc
import pandas as pd
import shutil
import tempfile
import tensorflow as tf
import mlflow.tensorflow

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]


def load_data(y_name="Species"):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path = tf.keras.utils.get_file(TRAIN_URL.split("/")[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split("/")[-1], TEST_URL)

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()



def main(argv):
    with mlflow.start_run():

        # Fetch the data

        try:

            # Since the model was automatically logged as an artifact (more specifically
            # a MLflow Model), we don't need to use saved_estimator_path to load back the model.
            # MLflow takes care of it!
            print(mlflow.get_artifact_uri("model"))
            m='/root/test/mlflow/tensorflow/tf2/mlruns/0/11c4d74436264f689c534bbd7eb214f7/artifacts/model'
            pyfunc_model = pyfunc.load_model(m)

            predict_data = [[5.1, 3.3, 1.7, 0.5], [5.9, 3.0, 4.2, 1.5], [6.9, 3.1, 5.4, 2.1]]
            df = pd.DataFrame(
                data=predict_data,
                columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"],
            )

            print("df")
            print(df)
            # Predicting on the loaded Python Function and a DataFrame containing the
            # original data we predicted on.
            predict_df = pyfunc_model.predict(df)

            # Checking the PyFunc's predictions are the same as the original model's predictions.
            template = '\nOriginal prediction is "{}", reloaded prediction is "{}"'
            for  pred in predict_df["classes"]:
                class_id = predict_df["class_ids"][
                    predict_df.loc[predict_df["classes"] == pred].index[0]
                ]
                reloaded_label = SPECIES[class_id]
                #print(template.format(expec, reloaded_label))

        finally:
            print("done")

if __name__ == "__main__":
    main(sys.argv)
