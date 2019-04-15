import os
import torch
from torch import nn
import numpy as np
import random
import time
from resnet import resnet50
from triplet_loss import hard_sdtw_triplet
from torch.autograd import Variable

import cv2

import tensorflow as tf
import keras
from keras.layers import Lambda
from keras import Input
from keras import backend as K
K.set_image_dim_ordering('tf') 
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.engine import Model
from keras.layers import Lambda, Dense, Dropout, Flatten, BatchNormalization, AveragePooling2D,Activation,Conv2D,MaxPooling2D,GlobalMaxPooling2D,ZeroPadding2D,GlobalAveragePooling2D
from keras import layers


from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import plot_model, to_categorical, multi_gpu_model
from numpy.random import randint, shuffle, choice, permutation



batch_num=0
SN = 6 # the number of images in a class
PN = 12
input_shape=(384,128,3)


def mix_data_prepare(data_list_path, train_dir_path):
    class_img_labels = dict()
    class_cnt = -1
    last_label = -2
    last_type = ''
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img = line
            lbl = int(line.split('_')[0])
            img_type = line.split('.')[-1]
            if lbl != last_label or img_type != last_type:
                class_cnt = class_cnt + 1
                cur_list = list()
                class_img_labels[str(class_cnt)] = cur_list
            last_label = lbl
            last_type = img_type

            img = image.load_img(os.path.join(train_dir_path, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            class_img_labels[str(class_cnt)].append(img[0])
    return class_img_labels


def reid_data_prepare(data_list_path, train_dir_path):
    if 'mix' in data_list_path:
        return mix_data_prepare(data_list_path, train_dir_path)
    class_img_labels = dict()
    class_cnt = -1
    last_label = -2
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img = line
            lbl = int(line.split('_')[0])
            if lbl != last_label:
                class_cnt = class_cnt + 1
                cur_list = list()
                class_img_labels[str(class_cnt)] = cur_list
            last_label = lbl
            img = os.path.join(train_dir_path, img)
            class_img_labels[str(class_cnt)].append(img)

    return class_img_labels

def load_and_process(pre_image):
    img = cv2.imread(pre_image)
    img = cv2.resize(img, (input_shape[1], input_shape[0]))

    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255.0

    img = img - np.array([0.485, 0.456, 0.406])
    img = img/np.array([0.229, 0.224, 0.225])

    return img

#def random_crop(image, crop_size):
#    w=np.random.randint(256-crop_size)   
#    h=np.random.randint(256-crop_size)
#    return image[h:h+crop_size, w:w+crop_size,:]

def triplet_hard_generator(class_img_labels, batch_size, train=False):
    cur_epoch = 0
    pos_prop = 5
    global PN
    global SN
    global input_shape
    while True:
        pre_label = permutation(len(class_img_labels))[0:PN]
        #pre_label = randint(len(class_img_labels), size=PN)
        pre_images = list()
        for i in range(PN):
            #len_pre_label_i = len(class_img_labels[str(pre_label[i])])
            labels_i  = class_img_labels[str(pre_label[i])]
            if len(labels_i) >= SN:
                label_index = permutation(len(labels_i))[0:SN]
            else:
                label_index = randint(len(labels_i), size=SN)

            for j in range(SN):
                pre_image = labels_i[label_index[j]]
                '''img = image.load_img(pre_image, target_size=[input_shape[0], input_shape[1]])
                img = image.img_to_array(img)
                img = preprocess_input(img)
                cv2.imshow("pre", img)
                cv2.waitKey(1000)
                '''
                img = load_and_process(pre_image).astype(np.float32)
                

                
                #img=random_crop(img, 224)
                
                #img = preprocess_input(img)[0]
                
                #if random.random()>0.5:
                #    img = img[:,::-1,:]
                pre_images.append(img)
	#print(pre_label)
        label=np.array([pre_label for i in range(SN)])
        label=np.transpose(label).reshape(SN*PN,1)
        label=np.squeeze(label)
        #print(label)
        label = to_categorical(label, num_classes=len(class_img_labels))
        #print(label)
        cur_epoch += 1
        yield np.array(pre_images), label

def common_lr(epoch):
    epsil = 0.01
    gamma = 0.1
    if epoch < 40:
        lr= epsil
    elif epoch<60:
        lr = epsil*gamma #epsil * np.power(1e-3, (epoch-65)/(110-65))
    else:
        lr = epsil*gamma*gamma#1e-4 * np.power(1e-3, (epoch-120)/(160-120))
   # else:
   #     lr = epsil*gamma*gamma*gamma
    return lr

reduce_lr = LearningRateScheduler(common_lr)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        if epoch< 60:
            param_group['lr'] = 3e-4#$param_group['lr']*(0.1 ** (epoch // 30))
        elif epoch < 80:
            param_group['lr']=1e-4
        else:
            param_group['lr']=3e-5

def pair_tune(source_model_path, train_generator, tune_dataset, batch_size=72, num_classes=751):
    global PN
    global SN
    device=torch.device("cuda")
    model = resnet50(True)
    model.to(device)
    #model = torch.load("./source_market_model.h5")

    num_epochs = 110
    batch_size = PN*SN

    f=open("./log.txt", "w")
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=learning_rate, weight_decay=0.0005)
    downConv1 = nn.Conv2d(2048, 1024, 1).to(device)
    downConv2 = nn.Conv2d(1024, 128, 1).to(device)
    bn = nn.BatchNorm2d(1024).to(device)
    bn2 = nn.BatchNorm2d(128).to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs ))
        since = time.time()

        adjust_learning_rate(optimizer, epoch)

        running_loss = 0.0
        model.train()
        # Iterate over data.
        for i_ in range(16500 // batch_size + 1):
            data_=train_generator.__next__()    
            inputs=data_[0]
            labels = data_[1]
            #print(inputs.shape)
            inputs=np.transpose(inputs, (0,3,1,2))
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            inputs=Variable(inputs, requires_grad=True)
            labels=Variable(labels, requires_grad=True)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                outputs =  downConv1(outputs)
                outputs = bn(outputs)
                outputs =  downConv2(outputs)
                #outputs = bn2(outputs)
                #print(outputs.shape)
                loss = hard_sdtw_triplet(outputs, f)
		
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            if i_ > 16500 // batch_size +1-21:
                running_loss +=  loss.item()
            f.write('Loss: {:.4f}'.format(loss.item()) +"\n")
            f.flush()
            #print('Loss: {:.4f}'.format(loss.item()))
        print('Loss: {:.4f}'.format(running_loss/20))
            # statistics
            #running_loss += loss.item() * inputs.size(0)


        #epoch_loss = running_loss / inputs.size(0)

        #print('Loss: {:.4f}'.format(epoch_loss))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    f.close()
    torch.save(model, "./market_model.h5")
    return model



                  
   # model.compile(optimizer=Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, decay=0.0005),
   #              loss=triplet_hard_loss)
    #early_stopping = EarlyStopping(monitor='loss', patience=4)
    #auto_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
    #                            cooldown=0, min_lr=0)
    # save_model = ModelCheckpoint('resnet50-{epoch:02d}-{val_ctg_out_1_acc:.2f}.h5', period=2)
    
    



def pair_pretrain_on_dataset(source, project_path='.', dataset_parent='../dataset'):
    if source == 'market':
        train_list = project_path + '/dataset/market_train.list'
        train_dir = dataset_parent + '/Market-1501-v15.09.15/bounding_box_train'
        class_count = 751
    elif source == 'markets1':
        train_list = project_path + '/dataset/markets1_train.list'
        train_dir = dataset_parent + '/markets1'
        class_count = 751
    elif source == 'grid':
        train_list = project_path + '/dataset/grid_train.list'
        train_dir = dataset_parent + '/grid_label'
        class_count = 250
    elif source == 'cuhk':
        train_list = project_path + '/dataset/cuhk_train.list'
        train_dir = dataset_parent + '/cuhk01'
        class_count = 971
    elif source == 'viper':
        train_list = project_path + '/dataset/viper_train.list'
        train_dir = dataset_parent + '/viper'
        class_count = 630
    elif source == 'duke':
        train_list = project_path + '/dataset/duke_train.list'
        train_dir = dataset_parent + '/DukeMTMC-reID/train'
        class_count = 702
    elif 'grid-cv' in source:
        cv_idx = int(source.split('-')[-1])
        train_list = project_path + '/dataset/grid-cv/%d.list' % cv_idx
        train_dir = dataset_parent + '/grid_train_probe_gallery/cross%d/train' % cv_idx
        class_count = 125
    elif 'mix' in source:
        train_list = project_path + '/dataset/mix.list'
        train_dir = dataset_parent + '/cuhk_grid_viper_mix'
        class_count = 250 + 971 + 630
    else:
        train_list = 'unknown'
        train_dir = 'unknown'
        class_count = -1
    class_img_labels = reid_data_prepare(train_list, train_dir)
    
    batch_size = SN*PN
    pair_tune(
        source + '_softmax_pretrain.h5',
        triplet_hard_generator(class_img_labels, batch_size=batch_size, train=True),
        source,
        batch_size=batch_size, num_classes=class_count
    )

if __name__ == '__main__':
    sources = ['market']
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #sources = ['cuhk', 'viper', 'market','duke']
    for source in sources:
        #softmax_pretrain_on_dataset(source,
        #                            project_path='/home/person/rank-reid-release',
        #                            dataset_parent='/home/person/dataset')
        pair_pretrain_on_dataset(source)
  

    # sources = ['viper']
    # for source in sources:
    #     # softmax_pretrain_on_dataset(source,
    #     #                             project_path='/home/person/rank-reid-release',
    #     #                             dataset_parent='/home/person/')
    #     pair_pretrain_on_dataset(source)
    # sources = ['grid-cv-%d' % i for i in range(10)]
    # for source in sources:
    #     softmax_pretrain_on_dataset(source,
    #                                 project_path='/home/person/rank-reid-release',
    #                                 dataset_parent='/home/person')
    #     pair_pretrain_on_dataset(source,
    #                              project_path='/home/person/rank-reid-release',
    #                              dataset_parent='/home/person')

