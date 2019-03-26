from __future__ import division, print_function, absolute_import

import os
from random import shuffle
import sys
import time
sys.path.append("./")

import torch
from  torch import nn
from torch.autograd import Variable
from resnet import resnet50

import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


def load_mix_data(LIST, TRAIN):
    images, labels = [], []
    with open(LIST, 'r') as f:
        last_label = -1
        label_cnt = -1
        last_type = ''
        for line in f:
            line = line.strip()
            img = line
            lbl = line.split('_')[0]
            cur_type = line.split('.')[-1]
            if last_label != lbl or last_type != cur_type:
                label_cnt += 1
            last_label = lbl
            last_type = cur_type
            img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            images.append(img[0])
            labels.append(label_cnt)

    img_cnt = len(labels)
    shuffle_idxes = range(img_cnt)
    shuffle(shuffle_idxes)
    shuffle_imgs = list()
    shuffle_labels = list()
    for idx in shuffle_idxes:
        shuffle_imgs.append(images[idx])
        shuffle_labels.append(labels[idx])
    images = np.array(shuffle_imgs)
    labels = to_categorical(shuffle_labels)
    return images, labels


def load_data(LIST, TRAIN):
    images, labels = [], []
    with open(LIST, 'r') as f:
        last_label = -1
        label_cnt = -1
        for line in f:
            line = line.strip()
            img = line
            lbl = line.split('_')[0]
            if last_label != lbl:
                label_cnt += 1
            last_label = lbl
            img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            images.append(img[0])
            labels.append(label_cnt)

    img_cnt = len(labels)
    shuffle_idxes = list(range(img_cnt))
    shuffle(shuffle_idxes)
    shuffle_imgs = list()
    shuffle_labels = list()
    for idx in shuffle_idxes:
        shuffle_imgs.append(images[idx])
        shuffle_labels.append(labels[idx])
    images = np.array(shuffle_imgs)
    #print(shuffle_labels)
    return images, shuffle_labels

def softmax_model_pretrain(train_list, train_dir, class_count, target_model_path):
    device=torch.device("cuda")
    model = resnet50(True)
    model.to(device)

    num_epochs = 40
    batch_size = 16

    images, labels = load_data(train_list, train_dir)
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        width_shift_range=0.2,  # 0.
        height_shift_range=0.2)
    train_generator = train_datagen.flow(images, labels, batch_size=batch_size)

    f=open("./log1.txt", "w")
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate)
    
    fc = nn.Linear(2048, 751).to(device)
    Dropout = nn.Dropout(p=0.5)
    criterion =  nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs ))
        since = time.time()

        #adjust_learning_rate(optimizer, epoch)

        running_loss = 0.0
        model.train()
        # Iterate over data.
        for i_ in range(16500 // batch_size +1):
            data_=train_generator.__next__()    
            inputs=data_[0]
            labels = data_[1]
            inputs=np.transpose(inputs, (0,3,1,2))
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            inputs=Variable(inputs, requires_grad=True)
            labels=Variable(labels).long()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                outputs = Dropout(outputs)
                outputs = fc(outputs)
                
                loss = criterion(outputs, labels)

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
    torch.save(model, "./source_market_model.h5")
    return model




def softmax_pretrain_on_dataset(source, project_path='./', dataset_parent='/home/tianhui/dataset'):
    if source == 'market':
        train_list = project_path + '/dataset/market_train.list'
        train_dir = dataset_parent + '/Market-1501-v15.09.15/bounding_box_train'
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
        train_dir = dataset_parent + '/underground_reid/cross%d/train' % cv_idx
        class_count = 125
    elif 'mix' in source:
        train_list = project_path + '/dataset/mix.list'
        train_dir = dataset_parent + '/cuhk_grid_viper_mix'
        class_count = 250 + 971 + 630
    else:
        train_list = 'unknown'
        train_dir = 'unknown'
        class_count = -1
    softmax_model_pretrain(train_list, train_dir, class_count, './'+source + '_softmax_pretrain.h5')


if __name__ == '__main__':
    # sources = ['market', 'grid', 'cuhk', 'viper']
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sources = ['market']
    for source in sources:
        softmax_pretrain_on_dataset(source)
