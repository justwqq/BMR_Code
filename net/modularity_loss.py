#This is the modularity loss
import itertools
import random
import torch
from scipy.spatial.distance import cosine
a1 = [29, 30, 41, 42, 71, 72, 73, 74, 75, 76, 81, 82, 83, 84, 87, 88]  # original index of ROIs in SN
a2 = [23, 24, 25, 26, 31, 32, 35, 36, 37, 38, 39, 40, 61, 62, 63, 64, 65, 66, 67, 68, 89, 90]  # DMN
a3 = [3, 4, 7, 8, 59, 60, 13, 14, 15, 16]  # CEN
list1 = torch.tensor(a1) - 1
list2 = torch.tensor(a2) - 1
list3 = torch.tensor(a3) - 1
m = 0.5
IDX1 = list(itertools.combinations(list1, 2))  # pairwise index in modularity1
IDX1 = random.choices(IDX1, k=int(len(IDX1) * m))
IDX2 = list(itertools.combinations(list2, 2))  # pairwise index in modularity2
IDX2 = random.choices(IDX2, k=int(len(IDX2) * m))
IDX3 = list(itertools.combinations(list3, 2))  # pairwise index in modularity3
IDX3 = random.choices(IDX3, k=int(len(IDX3) * m))
Idx_set = [IDX1, IDX2, IDX3]

def calculateloss(X):
    loss = 0
    for x in X:
        list2 = []
        for j in range(3):
            IDX = Idx_set[j]
            list1 = []
            for a, b in IDX:
                roi1 = x[a]  # (64,)
               # print(roi1.shape)
                roi2 = x[b]  # (64,)
                cos = -torch.cosine_similarity(roi1, roi2, dim=-1)
                list1.append(cos)
            loss1 = torch.sum(torch.stack(list1))
            list2.append(loss1)
        loss2 = torch.sum(torch.stack(list2))
        loss += loss2
    return loss
