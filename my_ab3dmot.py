import numpy as np
from box_calculator import box2_calculator
from AB3DMOT_libs.model import AB3DMOT


def pred_for_predestrian(pred_dicts):
    pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy().tolist()
    pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy().tolist()
    pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy().tolist()

    result = [[], [], []]
    for i in range(len(pred_boxes)):
        if pred_labels[i] == 2:
            result[0].append(pred_boxes[i])
            result[1].append(pred_labels[i])
            result[2].append(pred_scores[i])

    return result




def AB3DMOT_fun(pred_dicts, idx, mot_tracker):
    """pred_boxes = pred_dicts[0]['pred_boxes'][0].cpu().numpy()
    pred_labels = pred_dicts[0]['pred_labels'][0].cpu().numpy()
    pred_scores = pred_dicts[0]['pred_scores'][0].cpu().numpy()

    for i in range(1, pred_dicts[0]['pred_boxes'].cpu().numpy().shape[0]):
        if pred_dicts[0]['pred_scores'][i] > 0.2:
            print(pred_dicts[0]['pred_boxes'][i].cpu().numpy())
            # print(pred_dicts[0]['pred_boxes'][i].cpu())
            np.concatenate([pred_boxes], [pred_dicts[0]['pred_boxes'][i].cpu().numpy()], axis=0)
            np.concatenate([pred_labels], [pred_dicts[0]['pred_labels'][i].cpu().numpy()], axis=0)
            np.concatenate([pred_scores], [pred_dicts[0]['pred_scores'][i].cpu().numpy()], axis=0)


    print(pred_boxes, pred_labels)"""


    pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
    pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()

    box2 = box2_calculator(pred_dicts)
    inputs = []
    PI = np.pi
    #mot_tracker = AB3DMOT()

    for i in range(len(pred_labels)):
        if pred_labels[i] == 2 and pred_scores[i] > 0.3:
            rotate_y = pred_boxes[i, 6]  # figure rotate_y
            while rotate_y > PI:
                rotate_y -= 2* PI

            while rotate_y < -PI:
                rotate_y += 2* PI

            alpha = pred_boxes[i, 6] + np.arctan2(pred_boxes[i, 5] , pred_boxes[i, 3]) + 1.5 * PI # figure alpha
            while alpha > PI:
                alpha -= 2* PI

            while alpha < -PI:
                alpha += 2* PI

            inputs.append([idx, pred_labels[i], box2[i, 0], box2[i, 1], box2[i, 2], box2[i, 3], pred_scores[i]*10,\
                           pred_boxes[i, 3], pred_boxes[i, 4],pred_boxes[i, 5], pred_boxes[i, 0], pred_boxes[i, 1], pred_boxes[i, 2] ,rotate_y, alpha]) # 输入格式转化为AB3DMOT对应的格式


    inputs = np.array(inputs)
    ori_array = inputs[:, -1].reshape(-1, 1)
    other_array = inputs[:, 1:7].reshape(-1, 6)
    additional_info = np.concatenate((ori_array, other_array), axis=1)
    dets = inputs[:, 7:14]

    dets_all = {'dets': dets, 'info': additional_info}

    trackers = mot_tracker.update(dets_all)
    idx_array = np.array(idx).repeat(trackers.shape[0]).reshape(-1, 1)
    # result = np.concatenate((idx_array, trackers[:, 7].reshape(-1, 1), trackers[:, 0:6].reshape(-1, 6)), axis=1)
    result = np.concatenate((idx_array, trackers[:, 7].reshape(-1, 1), trackers[:, 3:6], trackers[:, 0:3], trackers[:, 6].reshape(-1, 1)), axis=1)
    # result 为 frame ID x y z h w l组成的array
    return result


