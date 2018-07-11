# -*- coding: utf-8 -*-
import os


class Path(str):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def __truediv__(self, path):
        return Path(os.path.join(self.path, str(path)))

    def __rtruediv__(self, path):
        return Path(os.path.join(str(path), self.path))


class config:
    # config about path
    dataset_dir = Path(os.path.dirname(__file__)) / 'dataset'
    raw_data_name = 'raw_data.csv'
    result_dir = Path(os.path.dirname(__file__)) / 'train_result'
    ckpt_path = Path(result_dir) / 'ckpt_best.h5'
    log_path = Path(result_dir) / 'lstm_train_hist.txt'
    data_name = 'data.csv'
    label_name = 'label.csv'

    # config for data_handler
    filter_num = 2  # the minimum review number a user should have
    distance_mode = 1  # mode 0 is based on radius , 1 is based on Euclidean distance of two events

    # config for train
    seq_len = 160  # maximum review a user can be included in one training instance
    simple_mode_num = 1000  # simple model
    cell_num = 100  # lstm_cell number
    batch_size = 20
    epoch_size = 10
    val_split = 0.1  # spilt rate for validation data
