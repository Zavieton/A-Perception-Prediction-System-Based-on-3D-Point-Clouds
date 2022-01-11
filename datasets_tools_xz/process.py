import numpy as np
import os
from tqdm import tqdm
from colorama import Fore
import matplotlib.pyplot as plt
import scipy.interpolate as spi


def interpolation(datas):
    time_seq = np.linspace(0, 0.1*len(datas), len(datas))
    lock = True
    datas = np.array(datas)
    x = datas[:, 0]
    y = datas[:, 1]
    z = datas[:, 2]

    def inter3(x, time_seq):
        ix3 = np.linspace(time_seq[0], time_seq[-1], 5*len(time_seq))
        tck = spi.splrep(time_seq, x, k=3)
        iy3 = spi.splev(ix3, tck)

        if not lock:
            plt.plot(time_seq, x)
            plt.plot(ix3, iy3)
            plt.show()

        return iy3

    ix = inter3(x, time_seq)
    iy = inter3(y, time_seq)
    iz = inter3(z, time_seq)

    ans = np.array([ix, iy, iz])
    ans = ans.T
    return ans


def mean_blur(X, kernel_size = 3):
    lens = len(X)
    if(lens < kernel_size):
     return

    res = []
    for i in range((kernel_size//2)+1):
        res.append(sum(X[i:kernel_size-1+i])/kernel_size)

    for i in range((kernel_size//2)+1, len(X)-(kernel_size//2)):
        res.append((sum(res[-1-(kernel_size//2):-1]) + sum(X[i:i+(kernel_size//2)]))/kernel_size)

    for i in range(len(X)-(kernel_size//2), len(X)):
        a = X[i:len(X)-1]
        b = res[len(res)-(kernel_size-(len(X)-i))-1: len(res)-1]
        res.append((sum(a) + sum(b))/(len(a) + len(b)))

    return res


def show_distribution(path):
    distribution = []
    g = os.walk(path)
    for path, _, file_list in g:
        lens = len(file_list)
        for i in tqdm(range(lens), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE)):
            cur_path = os.path.join(path, file_list[i])
            count = get_length(cur_path)
            distribution.append(count)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(distribution, bins=100)
    ax.set_title("Length distribution")
    ax.set_xlabel("length")
    ax.set_ylabel("numbers")
    plt.show()


def save_data(datas, file_name):
    if os.path.exists(file_name):
        return

    for s in datas:
        with open(file_name, "a+") as fi:
            strs = str(s[0]) + " " + str(s[1]) + " " + str(s[2])
            fi.write('%s' % (strs))
        fi.close()


def get_length(path):
    count = 0
    with open(path, "r") as fi:
        for _, _ in enumerate(fi):
            count += 1
    fi.close()
    return count






