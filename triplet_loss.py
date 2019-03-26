import numpy as np
import time
from numba import jit
import torch as t
import torch.nn as nn
from torch.autograd import Variable
import os

SN = 4 # the number of images in a class
PN = 16
relu = nn.ReLU(inplace=False)
device=t.device("cuda")
'''
#@jit(nopython = True)
def triplet_hard_loss(y_true, y_pred):
    global SN  # the number of images in a class
    global PN  # the number of class
    feat_num = SN*PN # images num
    #y=np.linalg.norm(y_pred, axis=1, keepdims=True)
    y_ = np.expand_dims(np.sqrt(np.sum(np.square(y_pred),axis = 1)), axis=1)
    #print( np.mean(np.square(y-y_) , axis=1) )
    y_pred = y_pred/y_

    feat1 = np.tile(np.expand_dims(y_pred,axis = 0),(feat_num,1,1))
    feat2 = np.tile(np.expand_dims(y_pred,axis = 1),(1,feat_num,1))
    delta = feat1 - feat2
    dis_mat = np.sum(np.square(delta),axis = 2) + np.finfo(np.float32).eps # Avoid gradients becoming NAN
    dis_mat = np.sqrt(dis_mat)
    positive = dis_mat[0:SN,0:SN]
    negetive = dis_mat[0:SN,SN:]
    for i in range(1,PN):
        positive = np.concatenate([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
        if i != PN-1:
            negs = np.concatenate([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
        else:
            negs = dis_mat[i*SN:(i+1)*SN, 0:i*SN]#np.concatenate(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
        negetive = np.concatenate([negetive,negs],axis = 0)
    positive = np.max(positive,axis=1)
    negetive = np.min(negetive,axis=1)
    #positive = K.print_tensor(positive)
    a1 = 1.2
    loss = np.mean(np.maximum(0.0,positive-negetive+a1))
    return loss
'''

def triplet_hard_loss(y_true, y_pred):
    global SN  # the number of images in a class
    global PN  # the number of class
    feat_num = SN*PN # images num
    #y=np.linalg.norm(y_pred, axis=1, keepdims=True)
    y_ =t.sqrt(t.sum(y_pred**2, 1)).unsqueeze(1)
    #print( np.mean(np.square(y-y_) , axis=1) )
    y_pred = y_pred/y_

    feat1 = y_pred.unsqueeze(0).repeat(feat_num,1,1)
    feat2 = y_pred.unsqueeze(1).repeat(1,feat_num,1)
    delta = feat1 - feat2
    dis_mat = t.sum(delta**2, 2) + np.finfo(np.float32).eps # Avoid gradients becoming NAN
    dis_mat = t.sqrt(dis_mat)
    positive = dis_mat[0:SN,0:SN]
    negetive = dis_mat[0:SN,SN:]
    for i in range(1,PN):
        positive = t.cat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]], 0)
        if i != PN-1:
            negs = t.cat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]], 1)
        else:
            negs = dis_mat[i*SN:(i+1)*SN, 0:i*SN]#np.concatenate(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
        negetive = t.cat([negetive,negs], 0)
    positive = t.max(positive,1)[0]
    negetive = t.min(negetive,1)[0]
    #positive = K.print_tensor(positive)
    #a1 = t.Tensor([1.2]).to("cuda")
    x=relu(positive-negetive+1.2)
    loss = t.mean(x)
    return loss



def maximum(R,S):
    return relu(R-S)+S


def data_pre(feature):
    Num = PN*SN
    DIM = feature.shape[2]
    F_DIM  = feature.shape[1]
    Distance = t.zeros((Num*Num*F_DIM)).to(device)
    for i in range(Num):
        D = t.zeros((Num*F_DIM, DIM , DIM )).to(device)
        for j in range(Num):
            dis = t.bmm(feature[i,:,:,:],t.transpose(feature[j,:,:,:], 1,2) )
            D[j*F_DIM:(j+1)*F_DIM,:,:] = dis
        R = compute_softdtw(D, 0.1)
        Distance[Num*i*F_DIM: Num*(i+1)*F_DIM] = R
    Distance = t.reshape(Distance, (Num,Num, F_DIM ))
    Distance = t.mean(Distance,2)
    return Distance

def compute_softdtw(D, gamma):
    NUM = D.shape[0]
    DIM = D.shape[1]
    R = t.zeros((NUM, DIM + 1, DIM + 1)).to(device) +1e-8
    R[0, 0] = 0
    for j in range(1, DIM + 1):
        for i in range(1, DIM + 1):
            r0 = -R[:,i - 1, j - 1] / gamma
            r1 = -R[:,i - 1, j] / gamma
            r2 = -R[:,i, j - 1] / gamma
            
            rmax = maximum(maximum(r0, r1), r2)
            rsum = t.exp(r0 - rmax) + t.exp(r1 - rmax) + t.exp(r2 - rmax)
            softmin = - gamma * (t.log(rsum) + rmax)
            R[:,i, j] = D[:, i - 1, j - 1] + softmin
    return R[:,DIM,DIM]

def hard_sdtw_triplet(feature):
    Distance = data_pre(feature)
    Num = Distance.shape[0]
    negetive = t.zeros((Num, Num-SN )).to(device)
    positive = t.zeros((Num, SN)).to(device)
    for i in range(Num):
        negetive[i,:] = t.cat([Distance[i, 0:i//SN],Distance[i, i//SN+SN:]],0)
        positive[i,:] = Distance[i, i//SN:i//SN+SN]
    negetive = t.min(negetive,1)[0]
    positive = t.max(positive,1)[0]
    x=relu(positive-negetive+1.2)
    loss = t.mean(x)
    return loss



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']="1"
    since = time.time()
    y_pred = t.rand(PN*SN,5,12,4).to(device)
    y_pred=Variable(y_pred, requires_grad=True)
    loss = hard_sdtw_triplet(y_pred)

    #for i in range(64):
    #    R = compute_softdtw(y_pred, 0.1)

    print("time:", time.time()-since, loss)
    #loss= triplet_hard_loss(y_pred, y_pred) 
    #print(loss)



