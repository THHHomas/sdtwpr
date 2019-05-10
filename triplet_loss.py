import numpy as np
import time
from numba import jit
import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os

SN = 3 # the number of images in a class
PN = 18

relu = nn.ReLU(inplace=False)
device=t.device("cuda")
avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

def triplet_hard_loss(y_pred, f):
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
    dis_mat = t.sum(delta**2, 2)*1000000+float(np.finfo(np.float32).eps) # Avoid gradients becoming NAN
    dis_mat = t.sqrt(dis_mat)/1000
    
    positive = dis_mat[0:SN,0:SN]
    negative = dis_mat[0:SN,SN:]
    for i in range(1,PN):
        positive = t.cat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]], 0)
        if i != PN-1:
            negs = t.cat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]], 1)
        else:
            negs = dis_mat[i*SN:(i+1)*SN, 0:i*SN]#np.concatenate(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
        negative = t.cat([negative,negs], 0)
    positive = t.max(positive,1)[0]
    negative = t.min(negative,1)[0]
    #positive = K.print_tensor(positive)
    #a1 = t.Tensor([1.2]).to("cuda")
    #print(positive, negative)
    x=relu(positive-negative+0.6)
    loss = t.mean(x)
    f.write(str(loss))
    return loss



def maximum(R,S):
    return relu(R-S)+S


def data_pre(feature):
    feature = l2_norm(feature)
    Num = PN*SN
    DIM = feature.shape[2]
    F_DIM  = feature.shape[1]
    Distance = t.zeros((Num*Num*F_DIM)).to(device)
    for i in range(Num):
        D = t.zeros((Num*F_DIM, DIM , DIM )).to(device)
        for j in range(Num):
            dis =t.sqrt(float(np.finfo(np.float32).eps)+relu(2*(1- t.bmm(feature[i,:,:,:],t.transpose(feature[j,:,:,:], 1,2) ))))
            '''input1 = feature[i,:,:,:].unsqueeze(1)
            input2 = feature[j,:,:,:].unsqueeze(2)
            input1 = input1.repeat(1, 12,1,1)
            input2 = input2.repeat(1, 1,12,1)
            subRes = (input1 - input2)**2
            dis = t.sum(subRes, 3)+1e-16'''
            #print(feature[i,:,:,:].shape, t.norm(feature[i,:,:,:], dim =2, p=2))
            
            D[j*F_DIM:(j+1)*F_DIM,:,:] = dis

        #print(t.max(D), t.min(D))
        R = compute_localmatch(D)
        Distance[Num*i*F_DIM: Num*(i+1)*F_DIM] = R
    Distance = t.reshape(Distance, (Num,Num, F_DIM ))
    Distance = t.mean(Distance,2)
    
    #print(Distance.shape)
    return Distance/6

def compute_softdtw(D):
    ###batch DTW
    ### D with shape (N, W, W), N is the batch size, W is the length of squence
    gamma = 0.1
    NUM = D.shape[0]
    DIM = D.shape[1]
    R = t.zeros((NUM, DIM + 1, DIM + 1)).to(device) + 1e8

    R[:, 0, 0] = 0
    for j in range(1, DIM + 1):
        for i in range(1, DIM + 1):
            r0 = R[:,i - 1, j - 1] 
            r1 = R[:,i - 1, j] 
            r2 = R[:,i, j - 1]
            #rmin = -maximum(maximum(-r0, -r1), -r2)
            r0 = t.unsqueeze(r0, 0)
            r1 = t.unsqueeze(r1, 0)
            r2 = t.unsqueeze(r2, 0)
            r = t.cat((r0, r1, r2), 0)
            r = t.transpose(r, 0,1)
            r0_map = (r1>=r0)*(r2>=r0)
            r1_map = (r0>r1)*(r2>=r1)
            r2_map = (r0>r2)*(r1>r2)
            r_map = t.cat((r0_map, r1_map, r2_map), 0)
            r_map = t.transpose(r_map, 0,1)
            #rmin = -maximum(maximum(-r0, -r1), -r2)
            #rmax = maximum(maximum(r0, r1), r2)
            #rsum = t.exp(r0 - rmax) + t.exp(r1 - rmax) + t.exp(r2 - rmax)
            #softmin = - gamma * (t.log(rsum) + rmax)
            #print(r[0:10])
            pre_dis = t.masked_select(r, r_map)
            #print(pre_dis[0:10])
            #print("predis", pre_dis)
            #print("rmin", pre_dis)
            R[:,i, j] =  D[:, i-1, j-1] + pre_dis
    row = R[:, -1, 6:]
    col = R[:,6:, -1]
    rc = t.cat((row, col),1)
    #print("rcmin", t.min(rc,1))
    temp =  t.min(rc,1)
    res =temp[0] #
    #print("row",temp[1][0:30], temp[1].shape)
    #res = R[:, -1, -1]
    return res



def compute_localmatch(D):
    ###batch DTW
    ### D with shape (N, W, W), N is the batch size, W is the length of squence
    gamma = 0.1
    NUM = D.shape[0]
    DIM = D.shape[1]
    R = t.zeros((NUM, DIM + 1, DIM + 1)).to(device) + 1e8
    

    R_path_i = t.zeros((NUM, DIM+1 , DIM+1 ), dtype = t.uint8).to(device) 
    R_path_j = t.zeros((NUM, DIM +1, DIM + 1), dtype = t.uint8).to(device) 
    R[:, 0, 0] = 0
    R_path_i[:, 0, 0] = 1
    R_path_j[:, 0, 0] = 1
    for i in range(1, DIM + 1):
        R_path_i[:, i, 1] = i
        R_path_i[:, 1, i] = 1
        R_path_j[:, 1, i] = i
        R_path_j[:, i, 1] = 1

    R_path_i = R_path_i.reshape((R_path_i.shape[0], -1))
    R_path_j = R_path_i.reshape((R_path_j.shape[0], -1))

    tolabel = t.zeros((NUM,3), dtype=t.uint8).to(device) 
    tolabel[:,1] = tolabel[:,1] + 1
    tolabel[:,2] = tolabel[:,2] + 2

    nature = [x for x in range(NUM)]
    zeros = [0 for x in range(NUM)]
    for j in range(1, DIM + 1):
        for i in range(1, DIM + 1):
            index = [[[i - 1, j - 1],[i - 1, j],[i, j - 1]]]
            #index_i = [i, i-1, i-1]
            #index_j = [j-1, j-1, j]
            index = t.Tensor(index).to(device) 
            #index = index.repeat(NUM,1,1)
            r0 = R[:,i - 1, j - 1] 
            r1 = R[:,i - 1, j] 
            r2 = R[:,i, j - 1]
            #rmin = -maximum(maximum(-r0, -r1), -r2)
            r0 = t.unsqueeze(r0, 0)
            r1 = t.unsqueeze(r1, 0)
            r2 = t.unsqueeze(r2, 0)
            r = t.cat((r0, r1, r2), 0)
            r = t.transpose(r, 0,1)
            r0_map = (r1>=r0)*(r2>=r0)
            r1_map = (r0>r1)*(r2>=r1)
            r2_map = (r0>r2)*(r1>r2)
            r_map = t.cat((r0_map, r1_map, r2_map), 0)
            r_map = t.transpose(r_map, 0,1)

            label = t.sum(r_map*tolabel, 1).tolist()

            In = index[zeros, label]
            Index = In[:,0]*(DIM +1)+In[:,1]
            Index = Index.unsqueeze(1).long()

            #R_path_i.gather(0,In)
            #print(R_path_i.shape, t.index_select(R_path_i, 1, In[:,0]).shape)
            #In_a = In[:,0].tolist()
            #In_b = In[:,1].tolist()
            #print(label)
            #print(In[:][0])
            
            
            if i> 1 and j> 1:

                R_path_i[:,i*(DIM+1)+j] =  t.gather(R_path_i, 1, Index).squeeze()
                R_path_j[:,i*(DIM+1)+j] =  t.gather(R_path_j, 1, Index).squeeze()
                #R_path_j[:,i, j] = R_path_j[nature, In[:,0].tolist(), In[:,1].tolist()]

            #print(label)
            #rmin = -maximum(maximum(-r0, -r1), -r2)
            #rmax = maximum(maximum(r0, r1), r2)
            #rsum = t.exp(r0 - rmax) + t.exp(r1 - rmax) + t.exp(r2 - rmax)
            #softmin = - gamma * (t.log(rsum) + rmax)
            #print(r[0:10])
            pre_dis = t.masked_select(r, r_map)

            #print(pre_dis[0:10])
            #print("predis", pre_dis)
            #print("rmin", pre_dis)
            R[:,i, j] =  D[:, i-1, j-1] + pre_dis


    R_path_i = R_path_i.reshape((R_path_i.shape[0], (DIM+1),(DIM+1)))
    R_path_j = R_path_i.reshape((R_path_j.shape[0], (DIM+1),(DIM+1)))

    row = R[:, -1, 6:]
    col = R[:,6:, -1]
    rc = t.cat((row, col),1)

    row_i = R_path_i[:, -1, 6:]
    col_i = R_path_i[:,6:, -1]
    rc_i = t.cat((row_i, row_i),1)

    row_j = R_path_j[:, -1, 6:]
    col_j = R_path_j[:,6:, -1]
    rc_j = t.cat((row_j, row_j),1)
    #print("rcmin", t.min(rc,1))
    temp =  t.min(rc,1)
    res =temp[0] #
    return res



def compute_localmatch(D):
    ###batch DTW
    ### D with shape (N, W, W), N is the batch size, W is the length of squence
    gamma = 0.1
    NUM = D.shape[0]
    DIM = D.shape[1]
    R = t.zeros((NUM, DIM + 1, DIM + 1)).to(device) + 1e8
    R_path_i = t.zeros((NUM, DIM+1 , DIM+1 ), dtype = t.uint8).to(device) 
    R_path_j = t.zeros((NUM, DIM +1, DIM + 1), dtype = t.uint8).to(device) 

    R[:, 0, 0] = 0
    R_path_i[:, 0, 0] = 1
    R_path_j[:, 0, 0] = 1
    for i in range(1, DIM + 1):
        R_path_i[:, i, 1] = i
        R_path_i[:, 1, i] = 1
        R_path_j[:, 1, i] = i
        R_path_j[:, i, 1] = 1


    tolabel = t.zeros((NUM,3), dtype=t.uint8).to(device) 
    tolabel[:,1] = tolabel[:,1] + 1
    tolabel[:,2] = tolabel[:,2] + 2

    nature = [x for x in range(NUM)]
    zeros = [0 for x in range(NUM)]
    for j in range(1, DIM + 1):
        for i in range(1, DIM + 1):
            index = [[[i - 1, j - 1],[i - 1, j],[i, j - 1]]]
            #index_i = [i, i-1, i-1]
            #index_j = [j-1, j-1, j]
            index = t.Tensor(index)
            #index = index.repeat(NUM,1,1)
            r0 = R[:,i - 1, j - 1] 
            r1 = R[:,i - 1, j] 
            r2 = R[:,i, j - 1]
            #rmin = -maximum(maximum(-r0, -r1), -r2)
            r0 = t.unsqueeze(r0, 0)
            r1 = t.unsqueeze(r1, 0)
            r2 = t.unsqueeze(r2, 0)
            r = t.cat((r0, r1, r2), 0)
            r = t.transpose(r, 0,1)
            r0_map = (r1>=r0)*(r2>=r0)
            r1_map = (r0>r1)*(r2>=r1)
            r2_map = (r0>r2)*(r1>r2)
            r_map = t.cat((r0_map, r1_map, r2_map), 0)
            r_map = t.transpose(r_map, 0,1)

            label = t.sum(r_map*tolabel, 1).tolist()
            In = index[zeros, label]
            #print(label)
            #print(In[:][0])
            if i> 1 and j> 1:
                R_path_i[:,i,j] = R_path_i[nature, In[:,0].tolist(), In[:,1].tolist()]
                R_path_j[:,i,j] = R_path_j[nature, In[:,0].tolist(), In[:,1].tolist()]

            #print(label)
            #rmin = -maximum(maximum(-r0, -r1), -r2)
            #rmax = maximum(maximum(r0, r1), r2)
            #rsum = t.exp(r0 - rmax) + t.exp(r1 - rmax) + t.exp(r2 - rmax)
            #softmin = - gamma * (t.log(rsum) + rmax)
            #print(r[0:10])
            pre_dis = t.masked_select(r, r_map)

            #print(pre_dis[0:10])
            #print("predis", pre_dis)
            #print("rmin", pre_dis)
            R[:,i, j] =  D[:, i-1, j-1] + pre_dis

    row = R[:, -1, 6:]
    col = R[:,6:, -1]
    rc = t.cat((row, col),1)

    row_i = R_path_i[:, -1, 6:]
    col_i = R_path_i[:,6:, -1]
    rc_i = t.cat((row_i, row_i),1)

    row_j = R_path_j[:, -1, 6:]
    col_j = R_path_j[:,6:, -1]
    rc_j = t.cat((row_j, row_j),1)
    #print("rcmin", t.min(rc,1))
    temp =  t.min(rc,1)
    res =temp[0] #

    index = temp[1].tolist()

    index_i = (rc_i[nature, index]).tolist()
    index_j = (rc_j[nature, index]).tolist()
    #print(index_i)
    #print(index_j)
    res = res - R[nature, index_i, index_j]
=======

    index = temp[1].unsqueeze(1).long()
    #index = temp[1].unqueeze(1).long()

    index_i = t.gather(rc_i, 1, index).squeeze()   #(rc_i[nature, index]).tolist()
    index_j = t.gather(rc_j, 1, index).squeeze()  #(rc_j[nature, index]).tolist()
    #print(index_i)
    #print(index_j)
    R=R.reshape((R.shape[0], -1))
    index = (index_i*(DIM+1)+ index_j).unsqueeze(1).long()


    res = res - t.gather(R, 1, index).squeeze()#R[nature, index_i, index_j]
>>>>>>> c8146f968d53f2144ab88917aa69d7978eaf6424
    #print(res.shape, R.shape)
    #print(temp[1])
    #print("row",temp[1][0:30], temp[1].shape)
    #res = R[:, -1, -1]
    return res

'''
def compute_softdtw(D):
    gamma = 0.1
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
    return R[:,-2,-2]
'''
dropout = nn.Dropout(0.5)
def ec_distance(feature):

    feature = avgpool(feature)
    feature =feature.squeeze()
    y_ =t.sqrt(t.sum(feature**2, 1)).unsqueeze(1)
    #print( np.mean(np.square(y-y_) , axis=1) )
    feature = feature/y_
    #feature = dropout(feature)

    #print(t.sqrt(t.sum(feature**2, 1)))
    Num = feature.shape[0]
    input1 = feature.unsqueeze(0)
    input2 = feature.unsqueeze(1)

    input1 = input1.repeat(Num, 1,1)
    input2 = input2.repeat(1, Num, 1)
    subRes = (input1 - input2)**2
    res = t.sum(subRes, 2)/2
    return res

def l2_norm(feature):
    shape = feature.shape
    feature = t.reshape(feature,(-1,4))
    y_ =t.sqrt(t.sum(feature**2, 1)).unsqueeze(1)
    #print( np.mean(np.square(y-y_) , axis=1) )
    feature = feature/y_
    feature = t.reshape(feature,(shape[0], shape[1], shape[2], shape[3]))
    return feature

def hard_sdtw_triplet(feature, f):
    ### DTW distance
    
    
    ##print(feature[0,0:2,:,:])
    ##print(t.norm(feature, dim = 3, p=2)) 
    Distance = data_pre(feature)
    #
    #Distance = data_pre(feature)
    
    ##Euclidean Distance
    Distance = ec_distance(feature)
    ##print(Distance.shape, Distance)
    log = Distance[0,0:12].cpu().detach().numpy()
    f.write(str(log.tolist()))
    #print(Distance)
    Num = Distance.shape[0]
    negetive = t.zeros((Num, Num-SN )).to(device)
    positive = t.zeros((Num, SN)).to(device)
    for i in range(Num):
        negetive[i,:] = t.cat([Distance[i, 0:(i//SN)*SN],Distance[i, (i//SN)*SN+SN:]],0)
        positive[i,:] = Distance[i, (i//SN)*SN:(i//SN)*SN+SN]
    negetive = t.min(negetive,1)[0]
    positive = t.max(positive,1)[0]
    x=relu(positive-negetive+0.6)

    loss = t.mean(x)
    return loss

def calD(s1,s2):
    D = np.zeros((3, len(s1), len(s2)), dtype = np.float32)
    for i in range(len(s1)):
        for j in range(len(s2)):
            D[0,i,j] = abs(s1[i]-s2[j])
    D[1,:,:]=D[0,:,:]
    D[2,:,:]=D[0,:,:]
    return D


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    f=open("test.txt", "w")
    
    y_pred = t.ones(PN*SN,128,12,4).to(device)
    y_pred[0,0,0,0] = 1000
    y_pred=Variable(y_pred, requires_grad=True)
    since = time.time()
    loss = hard_sdtw_triplet(y_pred, f)
    '''s1 = [1,2,3,4,5,5,5,4]
    s2 = [1,3,5,4,4,4,4, 1]
    D = calD(s1,s2)
    print(D[0,:,:])
    #for i in range(64):
    #    R = compute_softdtw(y_pred, 0.1)
    D = Variable(t.Tensor(D) , requires_grad = True).to(device)
    res = compute_softdtw(D)
    print(res)
    '''
    print("time:", time.time()-since, loss)

    #loss= triplet_hard_loss(y_pred, y_pred) 
    #print(loss)



    