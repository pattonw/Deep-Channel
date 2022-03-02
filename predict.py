import click

from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import math

from pathlib import Path


physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


@click.command()
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    help="The h5 containing your model weights.",
)
@click.option(
    "--csv", type=click.Path(exists=True), help="The data you want to predict on."
)
def predict(model, csv):
    csv_file = csv
    csv_path = Path(csv)

    if csv_path.is_dir():
        for csv_file in csv_path.iterdir():
            batch_size = 256

            csv = pd.read_csv(csv, header=None)
            dataset = csv.values
            dataset = dataset.astype("float64")

            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            loaded_model = load_model(model)

            c = loaded_model.predict_classes(
                dataset[:, 1].reshape((-1, 1, 1, 1)),
                batch_size=batch_size,
                verbose=True,
            )

            idealized = pd.DataFrame(c, index=dataset[:, 0])
            idealized.to_csv(csv_path / f"{csv_file.name[:-4]}_idealized.csv")
    else:
        batch_size = 256

        csv = pd.read_csv(csv, header=None)
        dataset = csv.values
        dataset = dataset.astype("float64")

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        loaded_model = load_model(model)

        c = loaded_model.predict_classes(
            dataset[:, 1].reshape((-1, 1, 1, 1)), batch_size=batch_size, verbose=True
        )

        idealized = pd.DataFrame(c, index=dataset[:, 0])
        idealized.to_csv(csv_path/f"{csv_file.name[:-4]}_idealized.csv")


if __name__ == "__main__":
    predict()
