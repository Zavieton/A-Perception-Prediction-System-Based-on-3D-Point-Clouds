from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


def create_dataset(raw_data, gt):

    data = {}
    data_src = []
    data_seq_start = []
    data_frames = []
    data_dt = []
    data_peds = []

    inp, info = get_strided_data_clust(raw_data, gt, 1)

    dt_frames = info['frames']
    dt_seq_start = info['seq_start']
    dt_dataset = np.array([0]).repeat(inp.shape[0])
    dt_peds = info['peds']

    data_src.append(inp)
    data_seq_start.append(dt_seq_start)
    data_frames.append(dt_frames)
    data_dt.append(dt_dataset)
    data_peds.append(dt_peds)


    data['src'] = np.concatenate(data_src, 0)
    data['seq_start'] = np.concatenate(data_seq_start, 0)
    data['frames'] = np.concatenate(data_frames, 0)
    data['dataset'] = np.concatenate(data_dt, 0)
    data['peds'] = np.concatenate(data_peds, 0)

    mean = data['src'].mean((0, 1))
    std = data['src'].std((0, 1))

    return IndividualTfDataset(data, "train", mean, std), None

    return IndividualTfDataset(data, "train", mean, std), IndividualTfDataset(data_val, "validation", mean, std)


class IndividualTfDataset(Dataset):
    def __init__(self,data,name,mean,std):
        super(IndividualTfDataset,self).__init__()

        self.data=data
        self.name=name

        self.mean= mean
        self.std = std

    def __len__(self):
        return self.data['src'].shape[0]


    def __getitem__(self,index):
        return {'src':torch.Tensor(self.data['src'][index]),
                'frames':self.data['frames'][index],
                'seq_start':self.data['seq_start'][index],
                'dataset':self.data['dataset'][index],
                'peds': self.data['peds'][index]
                }



def get_strided_data_clust(dt, gt_size, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame = []
    ped_ids = []
    for p in ped:
        for i in range(1 + (raw_data[raw_data.ped == p].shape[0] - gt_size) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size, [0]].values.squeeze())
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size, 2:4].values)
            ped_ids.append(p)

    frames = np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids = np.stack(ped_ids)

    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0], 1, 2)), inp_te_np[:, 1:, 0:2] - inp_te_np[:, :-1, 0:2]),1)
    inp_norm = np.concatenate((inp_te_np, inp_speed), 2)
    inp_mean = np.zeros(4)
    inp_std = np.ones(4)

    return inp_norm[:, :gt_size], {'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),
                                   'frames': frames, 'peds': ped_ids}

