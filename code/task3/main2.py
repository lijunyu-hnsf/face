import cv2
import os
import numpy as np

Dtr=np.load("E:/renlianshibie/Data/data_tr.npy")
Ltr=np.load("E:/renlianshibie/Data/label_tr.npy")

def one_hot(l):
    b=list(set(l))
    n=len(b)
    hb=np.zeros((n,n))
    for i in range(n):
        hb[i][i]=1
    return b,hb

lst,lb=one_hot(Ltr)
ytr=[]

np.save("E:/renlianshibie/model/lst.npy",lst)
np.save("E:/renlianshibie/model/lb.npy",lb)