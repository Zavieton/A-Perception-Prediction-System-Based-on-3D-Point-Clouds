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


def process_interpolation(path): # yang tiao cha zhi
    def process(datas):
        rate = 5
        result = []

        while True:
            for i in range(1, len(datas)):
                if int(datas[i][0]) - int(datas[i-1][0]) != 1:
                    tmt = [int(datas[i-1][0])+1, datas[i][1], (datas[i-1][2]+datas[i][2])*0.5, (datas[i-1][3]+datas[i][3])*0.5]
                    datas.append(tmt)
                    datas = sorted(datas, key=(lambda x: x[0]), reverse=False)
                    continue
            break

        time_seq = np.linspace(datas[0][0], datas[-1][0], len(datas))
        datas = np.array(datas)

        x = datas[:, 2]
        y = datas[:, 3]

        def inter3(x, time_seq):
            ix3 = np.linspace(time_seq[0], time_seq[-1], rate * (len(time_seq)-1))
            if(len(time_seq) <= 3):
                tck = spi.splrep(time_seq, x, k=1)
            else:
                tck = spi.splrep(time_seq, x, k=3)

            iy3 = spi.splev(ix3, tck)

            return iy3

        ix = inter3(x, time_seq)
        iy = inter3(y, time_seq)

        lens = len(ix)
        t = np.linspace(min(datas[:, 0]), max(datas[:, 0]), lens+1)

        for i in range(len(ix)):
            result.append([t[i], datas[0][1], ix[i], iy[i]])

        result2 = np.array(result)

        xs = result2[:, 2]
        ys = result2[:, 3]

        xs = mean_blur(xs, kernel_size=3)
        ys = mean_blur(ys, kernel_size=3)

        res = []
        for i in range(lens):
            res.append([int(result[i][0] * 10), result[i][1], xs[i], ys[i]])
        return res

    with open(path, "r") as f:
        datass = []
        for i, l in enumerate(f):
            fields = l.split(" ")
            datass.append([int(fields[0]), float(fields[1]), float(fields[2]), float(fields[3])])
    f.close()

    datass = sorted(datass, key=(lambda x: x[1]), reverse=False)
    lists = [1.0]
    res = []
    tempt = []

    for each in datass:
        if each[1] in lists:
            tempt.append(each)
        else:
            lists.append(each[1])
            if(len(tempt) > 1):
                res += process(tempt)
            tempt = []
            tempt.append(each)

    for each in res:
        print(each)

    res = sorted(res, key=(lambda x: x[0]), reverse=False)
    return res



if __name__ == "__main__":
    data_save_dir = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_processed_xz"

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
                    strs += " "
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
                    strs += " "

                write_path = os.path.join(data_save_dir, c[0:4] + "_train.txt")
                with open(write_path, "a+") as fi:
                    fi.write('%s\n' % (strs))
                fi.close()
                savein = []
        f.close()
    print("****************training dataset finished!****************")


    data_save_dir_2 = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_totrain_xz"

    if os.path.exists(data_save_dir_2):
        shutil.rmtree(data_save_dir_2)
        os.mkdir(data_save_dir_2)
    else:
        os.mkdir(data_save_dir_2)

    if not os.path.exists(data_save_dir):
        raise Exception('Program ERR: Dir is not exist')
    else:
        print("Path is {}".format(data_save_dir))
        g = os.walk(data_save_dir)

        for path, _, file_list in g:
            lens = len(file_list)
            for i in tqdm(range(lens), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE)):
                cur_path = os.path.join(path, file_list[i])
                res = process_interpolation(cur_path)

                savein = []
                for k in range(len(res)):
                    strs = ""
                    for j in res[k]:
                        strs += str(j)
                        strs += "\t"

                    write_path = os.path.join(data_save_dir_2, file_list[i])
                    with open(write_path, "a+") as fi:
                        fi.write('%s\n' % (strs))
                    fi.close()
                    savein = []

