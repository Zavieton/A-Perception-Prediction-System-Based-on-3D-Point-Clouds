import os
from process import show_distribution
import shutil
from tqdm import tqdm
from colorama import Fore
import random
from process import save_data
import numpy as np
import math

def truncate(datas, save_path, file_name):
    l = [200, 400, 600, 800, 1000]
    save_name = os.path.join(save_path, file_name)

    for s in datas:
        with open(save_name, "a+") as fi:
            strs = str(s[0]) + " " + str(s[1]) + " " + str(s[2])
            fi.write('%s' % (strs))
        fi.close()

    if(len(datas) > 400):
        for each in l:
            count = 0
            epoch = 3
            total_len = len(datas)
            if total_len > each:
                while(total_len > (each*(count+1))):
                    for e in range(epoch):
                        random_seed = random.randint(-each // 5, each // 5)
                        if (each*(count+1)+random_seed) <= total_len:
                            strs = "_" + str(each*count) + "_" + str(each*(count+1)+random_seed)
                            path = os.path.join(save_path, file_name + strs)
                            save_data(datas[each*count:(each*(count+1)+random_seed)], path)
                        count += 1


def threshold_noisy(data, distance):
    for each in data:
        if each > distance/5:
            return False
        else:
            continue

    return True


def gauss_noisy(Datas, dir_path, file_name):
    epoch = 5
    for e in range(epoch):
        Datas = np.array(Datas)
        x = []
        y = []
        z = []
        for each in Datas[:, 0]:
            x.append(float(each))
        for each in Datas[:, 1]:
            y.append(float(each))
        for each in Datas[:, 2]:
            z.append(float(each))
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        xmax = max(x)
        ymax = max(y)
        zmax = max(z)
        xmin = min(x)
        ymin = min(y)
        zmin = min(z)

        distance = [(xmax - xmin), (ymax - ymin), (zmax - zmin)]
        distance = np.array(distance)
        # s :Standard Deviation
        s = (distance / 5) / (2 * 2.58)

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
        # z
        noisy_z = np.random.normal(0, s[2] * s[2], len(z))
        if not threshold_noisy(noisy_z, distance[2]):
            continue
        zz = z + noisy_z

        Datas = []

        for i in range(len(xx)):
            Datas.append([xx[i], yy[i], zz[i]])

        fname = file_name + "_" + "with_gauss_" + str(e)
        paths = os.path.join(dir_path, fname)
        for s in Datas:
            with open(paths, "a+") as fi:
                strs = str(s[0]) + " " + str(s[1]) + " " + str(s[2])
                fi.write('%s\n' % (strs))
            fi.close()




if __name__ == "__main__":
    # truncation expansion
    data_dir = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_enhance"

    save_path = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_truncation"

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    else:
        os.mkdir(save_path)

    show_d = True

    if show_d:
        show_distribution(data_dir)

    g = os.walk(data_dir)

    for path, _, file_list in g:
        lens = len(file_list)
        for i in tqdm(range(lens), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE)):
            cur_path = os.path.join(path, file_list[i])
            with open(cur_path, "r") as f:
                datas = []
                for _, l in enumerate(f):
                    fields = l.split(" ")
                    datas.append(fields)
                truncate(datas, save_path, file_list[i])

    if show_d:
        show_distribution(save_path)

    # gauss expansion
    save_path_gauss = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_withGauss"
    if os.path.exists(save_path_gauss):
        shutil.rmtree(save_path_gauss)
        os.mkdir(save_path_gauss)
    else:
        os.mkdir(save_path_gauss)

    r = os.walk(save_path)
    for path, _, file_list in r:
        lens = len(file_list)
        for i in tqdm(range(lens), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE)):
            cur_path = os.path.join(path, file_list[i])
            with open(cur_path, "r") as f:
                datas = []
                for _, l in enumerate(f):
                    fields = l.split(" ")
                    datas.append(fields)
            gauss_noisy(datas, save_path_gauss, file_list[i])


