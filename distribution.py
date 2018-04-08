import numpy as np
import os
from glob import glob

# paths = []
# for f in range(1,7):
#     p = '/mnt/raid/data/ni/twoears/scenes2018/train/fold' + str(f) +'/scene1'
#     path = glob(p + '/**/**/*.npz', recursive=True)
#     paths += path
# num_classes = 13
# total_frames = 0
# count_positive = 13*[0]
# for p in paths:
#     data = np.load(p)
#     y = data['y']
#     total_frames =total_frames+ y.shape[1]
#     for i in range(13):
#         count_positive[i] =count_positive[i]+(y[i,:]==1).sum()
# #
# print([x/sum(count_positive) for x in count_positive])
# print([1-x/sum(count_positive) for x in count_positive])
def get_weight(paths):
    num_classes = 13
    total_frames = 0
    count_positive = 13 * [0]
    count_negative = 13 * [0]
    for p in paths:
        if p == '/mnt/raid/data/ni/twoears/scenes2018/train/fold2/scene9/generalSoundsNI.footsteps.FOOTSTEPS-OUTDOOR_GEN-HDF-12316.wav.mat.npz':
            continue
        print(p)
        data = np.load(p)
        y = data['y']
        total_frames = total_frames + y.shape[1]
        for i in range(13):
            count_positive[i] = count_positive[i] + (y[i, :] == 1).sum()
            count_negative[i] = count_negative[i] + (y[i, :] == 0).sum()
    #
    # return [x / sum(count_positive) for x in count_positive]
    return count_positive,count_negative

result = []
for i in range(1,7):
    print(i)
    dir = '/mnt/raid/data/ni/twoears/scenes2018/train/fold' + str(i) + '/'
    folder_name = os.listdir(dir)
    for scene in folder_name:
        print(scene)
        p = dir + scene
        path = glob(p + '/*.npz')
        w_pos,w_neg = get_weight(path)
        row = [i,scene] + w_pos + w_neg
        result.append(row)
np.save('trainweight.npy',np.array(result))
# def get_weight(scene_list, cvid):
#     weight_dir = '/mnt/raid/data/ni/twoears/trainweight.npy'
#     #  folder, scene, w_postive, w_negative
#     w = np.load(weight_dir)
#     count_pos = count_neg = [0] * 13
#
#     for i in scene_list:
#         for j in w:
#             if j[0] == str(cvid) and j[1] == i:
#                 count_pos = [x + int(y) for x, y in zip(count_pos, j[2:15])]
#                 count_neg = [x + int(y) for x, y in zip(count_neg, j[15:28])]
#                 break
#     total = (sum(count_pos)+sum(count_neg))
#     return [x / total for x in count_pos],[x / total for x in count_neg]
# a = get_weight(['scene1'],1)