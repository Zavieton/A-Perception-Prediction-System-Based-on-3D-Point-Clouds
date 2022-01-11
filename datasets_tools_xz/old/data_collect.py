# author zavieton
# create 2021/11/03
# HIT AIUS KEY LAB

import os
import sys
import numpy as np
import pandas as pd
import shutil

def generate_str(datas):
    # from AB3DMOT
    strs = str(datas[10] + " " + datas[11] + " " + datas[12] + " " + datas[13] + " " + datas[14] + " " + datas[15])
    return strs

if __name__ == "__main__":
    threshold = 9

    data_save_dir = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_raw"

    train_ped = ["0017.txt", "0016.txt", "0012.txt"]

    test_ped = ["0017.txt", "0019.txt", "0020.txt", "0021.txt", "0022.txt", "0023.txt", "0024.txt", "0025.txt", ]

    data_dir_train = "/results/pointrcnn_Pedestrian_val/data"  # absolute path

    data_dir_test = "/results/pointrcnn_Pedestrian_test/data"

    if os.path.exists(data_save_dir):
        shutil.rmtree(data_save_dir)
        os.mkdir(data_save_dir)
    else:
        os.mkdir(data_save_dir)

    for c in test_ped:
        path = os.path.join(data_dir_test, c)
        print(path)
        with open(path, "r") as f:
            datas = []
            for i, l in enumerate(f):
                fields = l.split(" ")
                datas.append(fields)
            datas = sorted(datas, key=(lambda x: x[1]), reverse=True)
            savein = []
            for i in range(len(datas)):
                strs = generate_str(datas[i])
                savein.append(strs)

                if (i == 0 or datas[i][1] != datas[i-1][1]):
                    if len(savein) >= threshold:
                        write_path = c[0:4] + "_test_" + datas[i][1]
                        write_path = os.path.join(data_save_dir, write_path)
                        for s in savein:
                            with open(write_path, "a+") as fi:
                                fi.write('%s\n' % (s))
                            fi.close()
                    savein = []
        f.close()

    print("****************testing dataset finished!****************")

    for c in train_ped:
        path = os.path.join(data_dir_train, c)
        print(path)
        with open(path, "r") as f:
            datas = []
            for i, l in enumerate(f):
                fields = l.split(" ")
                datas.append(fields)
            datas = sorted(datas, key=(lambda x: x[1]), reverse=True)
            savein = []
            for i in range(len(datas)):
                strs = generate_str(datas[i])
                savein.append(strs)
                if (i == 0 or datas[i][1] != datas[i-1][1]):
                    if len(savein) >= threshold:
                        write_path = c[0:4] + "_train_" + datas[i][1]
                        write_path = os.path.join(data_save_dir, write_path)
                        for s in savein:
                            with open(write_path, "a+") as fi:
                                fi.write('%s\n' % (s))
                            fi.close()
                    savein = []
        f.close()

    print("****************training dataset finished!****************")

    print("****************finished****************")

