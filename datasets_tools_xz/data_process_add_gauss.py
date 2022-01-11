# author zavieton
# create 2021/11/17
# HIT AIUS KEY LAB

import os
import numpy as np
import shutil
from tqdm import tqdm
from colorama import Fore



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

            dir = {each[1]: max(v) + 1}
            hashmap.update(dir)
            ans.append([int(each[0]), hashmap[each[1]], float(each[2]), float(each[3])])

    return ans


def threshold_noisy(data, distance):
    for each in data:
        if each > distance/10:
            return False
        else:
            continue

    return True


def gauss_noisy(Datas, dir_path, file_name):
    epoch = 3
    for e in range(epoch):
        Datas = np.array(Datas)
        x = []
        y = []
        for each in Datas[:, 2]:
            x.append(float(each))
        for each in Datas[:, 3]:
            y.append(float(each))

        x = np.array(x)
        y = np.array(y)

        xmax = max(x)
        ymax = max(y)
        xmin = min(x)
        ymin = min(y)

        distance = [(xmax - xmin), (ymax - ymin)]
        distance = np.array(distance)
        # s :Standard Deviation
        s = (distance / 20) / (2 * 2.58)

        # x

        noisy_x = np.random.normal(0, s[0] * s[0], len(x))
        if not threshold_noisy(noisy_x, distance[0]):
            continue
        xx = x + noisy_x
        # y
        noisy_y = np.random.normal(0, s[1] * s[1], len(y))
        if not threshold_noisy(noisy_y, distance[1]):
            continue

        yy = y + noisy_y

        Datass = []

        for i in range(len(xx)):
            Datass.append([xx[i], yy[i]])

        fname = file_name[0:9] + "_" + "with_gauss_" + str(e) + ".txt"
        paths = os.path.join(dir_path, fname)
        for i in range(len(Datas)):
            with open(paths, "a+") as fi:
                strs = str(Datas[i][0]) + "\t" + str(Datas[i][1]) + "\t" + str(Datass[i][0]) + "\t" + str(Datass[i][1])
                fi.write('%s\n' % (strs))
            fi.close()


if __name__ == "__main__":
    data_save_dir = "/home/zavieton/3D/pedestrian_predict/DTP3D/data_fold/data_totrain_xz"
    data_save_dir_2 = "/home/zavieton/3D/pedestrian_predict/DTP3D/data_fold/data_totrain_with_gauss_xz"

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
                with open(cur_path, "r") as f:
                    datass = []
                    for _, l in enumerate(f):
                        fields = l.split("\t")
                        datass.append(fields)

                gauss_noisy(datass, data_save_dir_2, file_list[i])

