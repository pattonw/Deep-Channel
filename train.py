# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:10:07 2019

@author: ncelik34
"""


# Importing the libraries
import os
import numpy
import time
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Reshape,
    Activation,
    LSTM,
    BatchNormalization,
    TimeDistributed,
    Conv1D,
    MaxPooling1D,
)
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from tensorflow_addons.metrics import F1Score

import click

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def mcor(y_true, y_pred):
    # Matthews correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = tp * tn - fp * fn
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def make_roc(true, predicted):

    # roc curve plotting for multiple

    n_classesi = predicted.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classesi):
        fpr[i], tpr[i], _ = roc_curve(true[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    plt.plot(
        fpr[2],
        tpr[2],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[2],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    colors = ["aqua", "darkorange", "cornflowerblue", "red", "black", "yellow"]
    for i in range(n_classesi):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Zooom in View: Some extension of ROC to multi-class")
    plt.legend(loc="lower right")
    plt.show()


def step_decay(epoch):
    # Learning rate scheduler object
    initial_lrate = 0.001
    drop = 0.001
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def process_csv(csv: Path, batch_size=int):
    """
    Get train/test data from csv
    """
    # read data
    df = pd.read_csv(csv, header=None)
    dataset = df.values.astype("float64")

    # process data
    try:
        maxer = np.amax(dataset[:, 2])
    except IndexError:
        # This csv does not have any gt data.
        logger.warning(
            f"Trying to train on csv: {csv}, but there is no ground truth data. Skipping this file!"
        )
        return None
    maxeri = maxer.astype("int")
    maxchannels = maxeri
    idataset = dataset[:, 2].astype(int)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # train and test set split and reshape:
    train_size = int(len(dataset) * 0.80)
    modder = math.floor(train_size / batch_size)
    train_size = int(modder * batch_size)
    test_size = int(len(dataset) - train_size)
    modder = math.floor(test_size / batch_size)
    test_size = int(modder * batch_size)

    logger.debug(f"training set = {train_size}")
    logger.debug(f"test set = {test_size}")
    logger.debug(f"total length = {test_size + train_size}")

    x_train = dataset[:, 1]
    y_train = idataset[:]
    x_train = x_train.reshape((len(x_train), 1))
    y_train = y_train.reshape((len(y_train), 1))

    sm = SMOTE(sampling_strategy="auto", random_state=42)
    X_res, Y_res = sm.fit_sample(x_train, y_train)

    yy_res = Y_res.reshape((len(Y_res), 1))
    yy_res = to_categorical(yy_res, num_classes=maxchannels + 1)
    xx_res, yy_res = shuffle(X_res, yy_res)

    trainy_size = int(len(xx_res) * 0.80)
    modder = math.floor(trainy_size / batch_size)
    trainy_size = int(modder * batch_size)
    testy_size = int(len(xx_res) - trainy_size)
    modder = math.floor(testy_size / batch_size)
    testy_size = int(modder * batch_size)

    logger.debug("training set= ", trainy_size)
    logger.debug("test set =", testy_size)
    logger.debug("total length", testy_size + trainy_size)

    in_train, in_test = (
        xx_res[0:trainy_size, 0],
        xx_res[trainy_size : trainy_size + testy_size, 0],
    )
    target_train, target_test = (
        yy_res[0:trainy_size, :],
        yy_res[trainy_size : trainy_size + testy_size, :],
    )
    in_train = in_train.reshape(len(in_train), 1, 1, 1)
    in_test = in_test.reshape(len(in_test), 1, 1, 1)
    return in_train, target_train, in_test, target_test


@click.command()
@click.option(
    "-m",
    "--model-file",
    type=click.Path(exists=False),
    help="The h5 containing your model weights.",
)
@click.option(
    "--csv", type=click.Path(exists=True), help="The data you want to predict on."
)
@click.option(
    "-e", "--num-epochs", type=int, help="The number of epochs to train.", default=2
)
@click.option(
    "-mc", "--max-channels", type=int, help="The maximum number of channels.", default=1
)
def train(model_file, csv, num_epochs, max_channels):
    csv = Path(csv)
    model_file = Path(model_file)

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    batch_size = 256

    # Create model
    # TODO: Model is dependent on data (maxchannels). What should we do here?
    newmodel = Sequential()
    timestep = 1
    input_dim = 1
    newmodel.add(
        TimeDistributed(
            Conv1D(filters=64, kernel_size=1, activation="relu"),
            input_shape=(None, timestep, input_dim),
        )
    )
    newmodel.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    newmodel.add(TimeDistributed(Flatten()))

    newmodel.add(LSTM(256, activation="relu", return_sequences=True))
    newmodel.add(BatchNormalization())
    newmodel.add(Dropout(0.2))

    newmodel.add(LSTM(256, activation="relu", return_sequences=True))
    newmodel.add(BatchNormalization())
    newmodel.add(Dropout(0.2))

    newmodel.add(LSTM(256, activation="relu"))
    newmodel.add(BatchNormalization())
    newmodel.add(Dropout(0.2))

    newmodel.add(Dense(max_channels + 1))
    newmodel.add(Activation("softmax"))

    newmodel.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False),
        metrics=["accuracy", Precision(), Recall()],
    )

    logger.debug(newmodel.summary())

    lrate = LearningRateScheduler(step_decay)

    csvs = (
        [csv]
        if csv.is_file()
        else [sub_csv for sub_csv in csv.iterdir() if sub_csv.name.endswith(".csv")]
    )
    csv_data = [process_csv(csv, batch_size) for csv in csvs]
    csv_data = [x for x in csv_data if x is not None]
    for i in range(num_epochs):
        for in_train, target_train, in_test, target_test in csv_data:
            newmodel.fit(
                x=in_train,
                y=target_train,
                initial_epoch=i,
                epochs=i + 1,
                batch_size=batch_size,
                callbacks=[lrate],
                verbose=1,
                shuffle=False,
                validation_data=(in_test, target_test),
            )

    model_dir = model_file.parent
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    newmodel.save(model_file)


if __name__ == "__main__":
    train()
