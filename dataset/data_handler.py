# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
import numpy as np
from utils import haversine
import sys
import os
from config import config


def split_data_in_user_groups():
    """
    :param raw_data : raw data is in the form of  'shopId,userId,score,"date",filtered,latitude,longitude'

    :param count : for each user, this value is used to record the number of bad reviews.

    :param cur_label: this value decide whether this user is a review spammer.

    :param loc_arr: a record of this users location for different reviews

    :param cur_data : a list of values measuring the distance based on the distance mode

    :return: output two csv files  mode_x_data.csv and mode_x_label.csv based on the distance mode
    """

    raw_data = defaultdict(list)
    data = []
    label = []

    os.chdir(config.dataset_dir)
    with open(config.raw_data_name, "r", encoding="latin-1") as csvfile:
        csvfile.readline()
        read = csv.reader(csvfile)
        for i in read:
            i[3] = i[3].replace('-', '')
            raw_data[i[1]].append(i[1:])

    for key in raw_data:
        if len(raw_data[key]) > config.filter_num:
            sorted_data = sorted(raw_data[key], key=lambda x: int(x[2]))
            count = sum([int(item[3]) for item in sorted_data])
            cur_label = 0 if count / len(sorted_data) <= 0.5 else 1
            loc_arr = [[float(item[-2]), float(item[-1])] for item in sorted_data]

            if config.distance_mode == 0:
                center = list(zip(*loc_arr))
                center_point = np.array([np.mean(np.array(center[0])), np.mean(np.array(center[1]))])
                loc_arr = np.array(loc_arr)
                cur_data = [haversine(center_point, item) for item in loc_arr]
            else:
                loc_arr = np.array(loc_arr)
                cur_data = [haversine(loc_arr[i - 1], loc_arr[i]) for i in range(1, len(loc_arr))]
            data.append(cur_data)
            label.append(cur_label)

    path = ['mode', str(config.distance_mode), '_', ]
    file = ['data.csv', 'label.csv']
    with open('{0[0]}{0[1]}{0[2]}{1[0]}'.format(path, file), "w", newline="") as data_csv:

        data_writer = csv.writer(data_csv)
        for item in data:
            data_writer.writerow(item)
    with open('{0[0]}{0[1]}{0[2]}{1[1]}'.format(path, file), "w", newline="") as label_csv:

        label_writer = csv.writer(label_csv)
        label_writer.writerow(label)

    return 0


if __name__ == '__main__':
    split_data_in_user_groups()
    # os.chdir(config.dataset_dir)
    # with open(config.raw_data_name, "r", encoding="latin-1") as csvfile:
    #     print(csvfile.readline())
