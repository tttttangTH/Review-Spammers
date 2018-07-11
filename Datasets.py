# -*- coding: utf-8 -*-

import csv
from config import config
import os

import numpy as np
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self):
        """
        :param self.data : whole data

        :param self.label : whole label

        :param self.train_label :  label for train

        :param self.test_label :  label for test

        :param self.test_data :  data for test

        :param self.test_data :  data for train


        """

        self.data = self.load_data()
        self.label = self.load_label()

        self.data, self.label = self.data_shuffle(self.data, self.label)
        self.train_data, self.test_data, self.train_label, self.test_label = self.dataset_split()

        assert len(self.label) == len(self.data)
        assert len(self.train_label) == len(self.train_data)
        assert len(self.test_label) == len(self.test_data)

    def data_shuffle(self, data, label):
        # shuffle the whole dataset

        indices = np.random.permutation(data.shape[0])
        rand_data_x = data[indices]
        rand_data_y = label[indices]
        return rand_data_x, rand_data_y

    def load_data(self):
        """
         load whole data
        """

        cnt = 0
        data = []
        os.chdir(config.dataset_dir)
        path = ['mode', str(config.distance_mode), '_', ]
        file = ['data.csv', 'label.csv']
        with open('{0[0]}{0[1]}{0[2]}{1[0]}'.format(path, file), "r",
                  encoding="latin-1") as csvfile:

            read = csv.reader(csvfile)
            for item in read:
                cnt += 1
                # print (i)
                item = [float(i) for i in item]
                data.append(item)
                if config.simple_mode_num > 0 and cnt > config.simple_mode_num - 1:
                    break

        for item in data:

            if len(item) > config.seq_len:
                item[:] = item[:config.seq_len]

            while len(item) < config.seq_len:
                item.append(-1.0)

        # print (max_len)

        return np.array(data).reshape((-1, config.seq_len, 1))

    def load_label(self):
        """
        load whole label
        """

        label = []
        os.chdir(config.dataset_dir)
        path = ['mode', str(config.distance_mode), '_', ]
        file = ['data.csv', 'label.csv']
        with open('{0[0]}{0[1]}{0[2]}{1[1]}'.format(path, file), "r",
                  encoding="latin-1") as csvfile:
            read = csv.reader(csvfile)
            for i in read:
                label = i
            label = [float(i) for i in label]
            if config.simple_mode_num > 0:
                return np.array(label[:config.simple_mode_num])
            else:
                return np.array(label)

    def dataset_split(self):
        """
        split whole data into training data and test data
        """

        xtrain, xtest, ytrain, ytest = train_test_split(self.data,
                                                        self.label,
                                                        test_size=0.2,
                                                        random_state=0)
        return xtrain, xtest, ytrain, ytest
