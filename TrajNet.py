import baselineUtils
import torch.utils.data
from transformer.batch import subsequent_mask
import numpy as np
import torch
import individual_TF
import pandas as pd

device=torch.device("cuda")

batch_size = 512
layers = 6 
heads = 8
preds = 12  # 预测preds个输出
obs = 8

def cluster(data): #筛选，为了防止输入transformer的数据过多，每个ID只保留obs个检测结果
    data.sort_values(by=['frame', 'ped'], inplace=True)
    current_frame = max(data['frame'])
    current_peds = data[data['frame'] == current_frame]['ped']
    ans = pd.DataFrame(columns = ['frame', 'ped', 'x', 'y', 'z'])

    for each in current_peds:
        if len(data[data['ped'] == each]) >= obs:
            tempt = data[data['ped'] == each][-1-obs:-1]
            ans = pd.concat([ans, tempt], axis=0)
        else:
            pass

    ans.sort_values(by=['frame', 'ped'], inplace=True)
    return ans

def traject(raw_data, transformer_models):
    raw_data = pd.DataFrame(raw_data)
    raw_data.columns = ['frame', 'ped', 'x', 'y', 'z']

    # raw_data.sort_values(by=['frame', 'ped'], inplace=True)
    raw_datas = cluster(raw_data)
    raw_data = np.array(raw_datas)

    raw_data = np.concatenate((raw_data[:, 0:3].reshape(-1, 3), raw_data[:, 4].reshape(-1, 1)), axis=1)
    raw_data = pd.DataFrame(raw_data)
    raw_data.columns = ['frame', 'ped', 'x', 'y']

    # raw_data.sort_values(by=['frame', 'ped'], inplace=True)

    test_dataset, _ = baselineUtils.create_dataset(raw_data, obs)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    means = []
    stds = []
    for i in np.unique(test_dataset[:]['dataset']):
        ind = test_dataset[:]['dataset'] == i
        means.append(test_dataset[:]['src'][ind, 1:, 2:4].mean((0, 1)))
        stds.append(test_dataset[:]['src'][ind, 1:, 2:4].std((0, 1)))

    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)


    with torch.no_grad():
        transformer_models.eval()
        pr = []
        inp_ = []
        peds = []
        frames = []

        for id_b, batch in enumerate(test_dl):
            inp_.append(batch['src'])
            frames.append(batch['frames'])
            peds.append(batch['peds'])

            inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
            src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(device)
            dec_inp = start_of_seq

            for i in range(preds):
                trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                out = transformer_models(inp, dec_inp, src_att, trg_att)
                dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

            preds_tr_b = (dec_inp[:, 1:, 0:2] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + batch['src'][:, -1:,0:2].cpu().numpy()
            pr.append(preds_tr_b)

        peds = np.concatenate(peds, 0) # peds为预测的行人ID
        pr = np.concatenate(pr, 0) # pr为预测结果，此时的预测结果只包括xz
        results = []
        for i in range(len(peds)):
            means = np.average(raw_datas[raw_datas['ped'] == peds[i]]['y'])
            results.append(np.concatenate((pr[i][:, 0].reshape(-1, 1), means.repeat(pr[i].shape[0]).reshape(-1, 1), pr[i][:, 1].reshape(-1,1)), axis=1).tolist()) #加入y值，考虑到y几乎不变，取预测的y均为输入y的平均值

        results = np.array(results)
        # print(results)
        return results, peds #返回预测的xyz， 和行人ID
        # peds = np.concatenate(peds, 0)
        # frames = np.concatenate(frames, 0)
        # pr = np.concatenate(pr, 0)

def traject2(raw_data, transformer_models):
    raw_data = pd.DataFrame(raw_data)
    raw_data.columns = ['frame', 'ped', 'x', 'y', 'z']

    # raw_data.sort_values(by=['frame', 'ped'], inplace=True)
    raw_datas = cluster(raw_data)
    raw_data = np.array(raw_datas)

    # raw_data = np.concatenate((raw_data[:, 0:3].reshape(-1, 3), raw_data[:, 4].reshape(-1, 1)), axis=1)
    raw_data = raw_data[:, 0:4]
    raw_data = pd.DataFrame(raw_data)
    raw_data.columns = ['frame', 'ped', 'x', 'y']

    # raw_data.sort_values(by=['frame', 'ped'], inplace=True)

    test_dataset, _ = baselineUtils.create_dataset(raw_data, obs)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    means = []
    stds = []
    for i in np.unique(test_dataset[:]['dataset']):
        ind = test_dataset[:]['dataset'] == i
        means.append(test_dataset[:]['src'][ind, 1:, 2:4].mean((0, 1)))
        stds.append(test_dataset[:]['src'][ind, 1:, 2:4].std((0, 1)))

    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)


    with torch.no_grad():
        transformer_models.eval()
        pr = []
        inp_ = []
        peds = []
        frames = []

        for id_b, batch in enumerate(test_dl):
            inp_.append(batch['src'])
            frames.append(batch['frames'])
            peds.append(batch['peds'])

            inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
            src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(device)
            dec_inp = start_of_seq

            for i in range(preds):
                trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                out = transformer_models(inp, dec_inp, src_att, trg_att)
                dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

            preds_tr_b = (dec_inp[:, 1:, 0:2] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + batch['src'][:, -1:,0:2].cpu().numpy()
            pr.append(preds_tr_b)

        peds = np.concatenate(peds, 0) # peds为预测的行人ID
        pr = np.concatenate(pr, 0) # pr为预测结果，此时的预测结果只包括xz
        results = []
        for i in range(len(peds)):
            means = np.average(raw_datas[raw_datas['ped'] == peds[i]]['z'])
            results.append(np.concatenate((pr[i][:, 0].reshape(-1, 1), means.repeat(pr[i].shape[0]).reshape(-1, 1), pr[i][:, 1].reshape(-1,1)), axis=1).tolist()) #加入y值，考虑到y几乎不变，取预测的y均为输入y的平均值

        results = np.array(results)
        # print(results)
        return results, peds # 返回预测的xyz， 和行人ID

