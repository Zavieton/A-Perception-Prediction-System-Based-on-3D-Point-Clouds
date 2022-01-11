# author zavieton
# create 2021/11/03
# HIT AIUS KEY LAB

import os
import sys
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from colorama import Fore
from process import interpolation, mean_blur
import scipy.interpolate as spi

def generate_str(datas):
    # from AB3DMOT
    res = []
    for d in datas:
        res.append([d[0], d[1], d[13], d[15]])

    hashmap = {datas[0][1]: 1}
    ans = []

    for each in res:
        if each[1] in hashmap:
            ans.append([int(each[0]), hashmap[each[1]], float(each[2]), float(each[3])])
        else:
            v = []
            for value in hashmap.values():
                v.append(int(value))

            dir = {each[1]: max(v)+1}
            hashmap.update(dir)
            ans.append([int(each[0]), hashmap[each[1]], float(each[2]), float(each[3])])

    return ans



if __name__ == "__main__":
    data_save_dir = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_withoutInter_totrain_xz"

    train_ped = ["0017.txt", "0016.txt", "0012.txt"]
    test_ped = ["0017.txt", "0019.txt", "0020.txt", "0021.txt", "0022.txt", "0023.txt", "0024.txt", "0025.txt", ]

    data_dir_train = "/home/zavieton/3D/pedestrian_predict/AB3DMOT/results/pointrcnn_Pedestrian_val/data"  # absolute path
    data_dir_test = "/home/zavieton/3D/pedestrian_predict/AB3DMOT/results/pointrcnn_Pedestrian_test/data"

    if os.path.exists(data_save_dir):
        shutil.rmtree(data_save_dir)
        os.mkdir(data_save_dir)
    else:
        os.mkdir(data_save_dir)

    for c in test_ped:
        path = os.path.join(data_dir_test, c)
        with open(path, "r") as f:
            datas = []
            for i, l in enumerate(f):
                fields = l.split(" ")
                datas.append(fields)
            savein = []
            DATA = generate_str(datas)
            for i in range(len(DATA)):
                strs = ""
                for j in DATA[i]:
                    strs += str(j)
                    strs += "\t"
                write_path = os.path.join(data_save_dir, c[0:4] + "_test.txt")
                with open(write_path, "a+") as fi:
                    fi.write('%s\n' % (strs))
                fi.close()
                savein = []
        f.close()
    print("****************testing dataset finished!****************")

    for c in train_ped:
        path = os.path.join(data_dir_train, c)
        with open(path, "r") as f:
            datas = []
            for i, l in enumerate(f):
                fields = l.split(" ")
                datas.append(fields)
            savein = []
            DATA = generate_str(datas)
            for i in range(len(DATA)):
                strs = ""
                for j in DATA[i]:
                    strs += str(j)
                    strs += "\t"

                write_path = os.path.join(data_save_dir, c[0:4] + "_train.txt")
                with open(write_path, "a+") as fi:
                    fi.write('%s\n' % (strs))
                fi.close()
                savein = []
        f.close()
    print("****************training dataset finished!****************")

