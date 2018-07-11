# -*- coding : utf-8 -*-
from keras.layers import Masking, Dense, Input, BatchNormalization, Dropout, LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from callback import Metrics
import Datasets
from config import config
import numpy as np
import tensorflow as tf
import keras.backend as K


def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    """

    :param y_true: true label
    :param y_pred: predict label
    :param threshold:  threshold  for approximation
    :return: acc of negative class labels
    """

    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N


def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / P


def lstm_model(seq_len, cell_num):
    inputs = Input(shape=(seq_len, 1))
    mask = Masking(mask_value=-1.0,
                   input_shape=(seq_len, 1,))(inputs)
    lstm_out = LSTM(cell_num,
                    dropout_W=0.2,
                    dropout_U=0.2,
                    input_shape=(seq_len, 1,))(mask)
    dropout = Dropout(0.2)(lstm_out)
    bn = BatchNormalization()(dropout)
    predictions = Dense(1,
                        activation="sigmoid")(bn)

    model = Model(inputs=inputs,
                  outputs=predictions)
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy", ]
                  )

    model.summary()
    return model


def train():
    model = lstm_model(config.seq_len, config.cell_num)

    # callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(config.ckpt_path, save_best_only=True, save_weights_only=True)
    metrics = Metrics()

    dataset = Datasets.Dataset()

    hist = model.fit(dataset.train_data,
                     dataset.train_label,
                     batch_size=config.batch_size,
                     epochs=config.epoch_size,
                     shuffle=True,
                     validation_split=config.val_split,
                     callbacks=[early_stopping, model_checkpoint, metrics])

    with open(config.log_path, 'w') as f:
        f.write(str(hist.history) + '\n')
        f.write(str(metrics.val_f1s) + '\n')
        f.write(str(metrics.val_precisions) + '\n')
        f.write(str(metrics.val_recalls) + '\n')

    score, acc = model.evaluate(dataset.test_data,
                                dataset.test_label,
                                batch_size=config.batch_size)
    print(score, acc)


train()
