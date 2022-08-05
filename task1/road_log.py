# import matplotlib.pyplot as plt
import numpy as np

train_set = np.load((r'E:\音学\code\Accompaniment-Generation\task1\sketchnet_stage1_log.npy'),allow_pickle = True)
l1=[1,2]
l2=[3,4]
l3=l1.extend(l2)
print(l1)
# print(train_set.shape)
# for i in range(len(train_set)):
#     print(train_set[i][0])
#     print(train_set[i][1])
