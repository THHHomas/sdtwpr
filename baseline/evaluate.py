from __future__ import division, print_function, absolute_import

import os

import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image

from utils.file_helper import write, safe_remove


def extract_info(dir_path):
    infos = []
    for image_name in sorted(os.listdir(dir_path)):
        if '.jpg' not in image_name:
            continue
        if 's' in image_name or 'f' in image_name:
            # market && duke
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        infos.append((person, camera))

    return infos

def get_augmentation_batch(image):
    #Resize it correctly, as needed by the test time augmentation.
    #im_mean = np.asarray([103.0626238, 115.90288257, 123.15163084], dtype=np.float32)
    #image = cv2.resize(image, (224+32, 224+32))

    #Change into CHW format
    #image = np.rollaxis(image,2)
    image=np.squeeze(image)
    #Setup storage for the batch
    #print("image shape:", image.shape)
    batch = np.zeros((2, 384, 128, 3), dtype=np.float32)

    #Four corner crops and the center crop
    #batch[0] = image[16:-16, 8:-8,:]    #Center crop
    #batch[1] = image[ :-32, :-16, :] #Top left
    #batch[2] = image[ :-32, 16:, :]    #Top right
    #batch[3] = image[32:, :-16,:] #Bottom left
    #batch[4] = image[32:, 16:, :]    #Bottom right

    #Flipping
    #batch[5:] = batch[:5,:,::-1,:]

    #Subtract the mean
    #batch = batch-im_mean[None,:,None,None]
    batch[0]=image
    batch[1]=image[:,::-1,:]
    return batch

def extract_feature(dir_path, net):
    print("in extract feature:----------------------------------------------------------------- ")
    features = []
    infos = []
    for image_name in sorted(os.listdir(dir_path)):
        if '.jpg' not in image_name:
            continue
        if 'f' in image_name or 's' in image_name:
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        image_path = os.path.join(dir_path, image_name)
        img = image.load_img(image_path, target_size=(384, 128))
        x = image.img_to_array(img)
        
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = get_augmentation_batch(x)
        #print("x.shpae: ",x.shape)
        feature = net.predict(x)
        #feature[0]=np.squeeze(feature[0])
        #print(feature[0].shape, feature[1].shape)
        #print(len(feature), type(feature))
        #print(feature[0].shape)
        #feature = np.concatenate(feature,axis = 1)
        #print(feature.shape)
        feature = np.mean(feature, axis=0)
        features.append(np.squeeze(feature))
        infos.append((person, camera))
    return features, infos


def similarity_matrix(query_f, test_f):
    # Tensorflow graph
    # use GPU to calculate the similarity matrix
    query_t = tf.placeholder(tf.float32, (None, None))
    test_t = tf.placeholder(tf.float32, (None, None))
    query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
    test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
    tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    result = sess.run(tensor, {query_t: query_f, test_t: test_f})
    tf.reset_default_graph()
    # descend
    return result


def sort_similarity(query_f, test_f):
    result = similarity_matrix(query_f, test_f)
    result_argsort = np.argsort(-result, axis=1)
    return result, result_argsort


def similarity_matrix_mul(query_f, test_f):
    # Tensorflow graph
    # use GPU to calculate the similarity matrix
    query_t = tf.placeholder(tf.float32, (None, None))
    test_t = tf.placeholder(tf.float32, (None, None))
    query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
    test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
    tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


    query_f = np.array(query_f)
    test_f = np.array(test_f)
    #result = sess.run(tensor, {query_t: query_f, test_t: test_f})
    #tf.reset_default_graph()
    
    q0 = query_f[:,0:2048]
    q1 = query_f[:,2048:3072]
    q2 = query_f[:,3072:4096]
    q3 = query_f[:,4096:5120]

    t0 = test_f[:,0:2048]
    t1 = test_f[:,2048:3072]
    t2 = test_f[:,3072:4096]
    t3 = test_f[:,4096:5120]
    
    r0 = sess.run(tensor, {query_t: q0, test_t: t0})
    tf.reset_default_graph()
    r1 = sess.run(tensor, {query_t: q1, test_t: t1})
    tf.reset_default_graph()
    r2 = sess.run(tensor, {query_t: q2, test_t: t2})
    tf.reset_default_graph()
    r3 = sess.run(tensor, {query_t: q3, test_t: t3})
    tf.reset_default_graph()
    result  =  r0+0.2*r1+0.5*r2+0.2*r3

    print(result.shape)
    # descend
    return result

def sort_similarity_mul(query_f, test_f):
    result = similarity_matrix(query_f, test_f)
    result_argsort = np.argsort(-result, axis=1)
    print("get:", result)
    return result, result_argsort

def map_rank_quick_eval(query_info, test_info, result_argsort):
    # much more faster than hehefan's evaluation
    match = []
    junk = []
    QUERY_NUM = len(query_info)

    for q_index, (qp, qc) in enumerate(query_info):
        tmp_match = []
        tmp_junk = []
        for t_index in range(len(test_info)):
            p_t_idx = result_argsort[q_index][t_index]
            p_info = test_info[int(p_t_idx)]

            tp = p_info[0]
            tc = p_info[1]
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
            elif tp == qp or tp == -1:
                tmp_junk.append(t_index)
        match.append(tmp_match)
        junk.append(tmp_junk)

    rank_1 = 0.0
    mAP = 0.0
    rank1_list = list()
    for idx in range(len(query_info)):
        if idx % 100 == 0:
            print('evaluate img %d' % idx)
        recall = 0.0
        precision = 1.0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        ig_cnt = 0
        for ig in IGNORE:
            if ig < YES[0]:
                ig_cnt += 1
            else:
                break
        if ig_cnt >= YES[0]:
            rank_1 += 1
            rank1_list.append(1)
        else:
            rank1_list.append(0)

        for i, k in enumerate(YES):
            ig_cnt = 0
            for ig in IGNORE:
                if ig < k:
                    ig_cnt += 1
                else:
                    break
            cnt = k + 1 - ig_cnt
            hit = i + 1
            tmp_recall = hit / len(YES)
            tmp_precision = hit / cnt
            ap = ap + (tmp_recall - recall) * ((precision + tmp_precision) / 2)
            recall = tmp_recall
            precision = tmp_precision

        mAP += ap
    rank1_acc = rank_1 / QUERY_NUM
    mAP = mAP / QUERY_NUM
    print('Rank 1:\t%f' % rank1_acc)
    print('mAP:\t%f' % mAP)
    np.savetxt('rank_1.log', np.array(rank1_list), fmt='%d')
    return rank1_acc, mAP


def train_predict(net, train_path, pid_path, score_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    train_f, test_info = extract_feature(train_path, net)
    result, result_argsort = sort_similarity(train_f, train_f)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    # ignore top1 because it's the origin image
    np.savetxt(score_path, result[:, 1:], fmt='%.4f')
    np.savetxt(pid_path, result_argsort[:, 1:], fmt='%d')
    return result


def test_predict(net, probe_path, gallery_path, pid_path, score_path):
    #net = Model(inputs=[net.input], outputs=[net.output])
    #print("output shape:", net.output.shape)
    test_f, test_info = extract_feature(gallery_path, net)  ###19732
    query_f, query_info = extract_feature(probe_path, net)  ##3368
    result, result_argsort = sort_similarity(query_f, test_f)
    print("after result:", result)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    safe_remove(pid_path)
    safe_remove(score_path)
    np.savetxt(pid_path, result_argsort, fmt='%d')
    np.savetxt(score_path, result, fmt='%.4f')


def market_result_eval(predict_path, log_path='market_result_eval.log', TEST='Market-1501/test',
                       QUERY='Market-1501/probe'):
    res = np.genfromtxt(predict_path, delimiter=' ')
    print('predict info get, extract gallery info start')
    test_info = extract_info(TEST)
    
    print('extract probe info start')
    query_info = extract_info(QUERY)
    print('start evaluate map and rank acc')
    rank1, mAP = map_rank_quick_eval(query_info, test_info, res)
    write(log_path, predict_path + '\n')
    write(log_path, '%f\t%f\n' % (rank1, mAP))


def grid_result_eval(predict_path, log_path='grid_eval.log'):
    pids4probes = np.genfromtxt(predict_path, delimiter=' ')
    probe_shoot = [0, 0, 0, 0, 0]
    for i, pids in enumerate(pids4probes):
        for j, pid in enumerate(pids):
            if pid - i == 775:
                if j == 0:
                    for k in range(5):
                        probe_shoot[k] += 1
                elif j < 5:
                    for k in range(1, 5):
                        probe_shoot[k] += 1
                elif j < 10:
                    for k in range(2, 5):
                        probe_shoot[k] += 1
                elif j < 20:
                    for k in range(3, 5):
                        probe_shoot[k] += 1
                elif j < 50:
                    for k in range(4, 5):
                        probe_shoot[k] += 1
                break
    probe_acc = [shoot / len(pids4probes) for shoot in probe_shoot]
    write(log_path, predict_path + '\n')
    write(log_path, '%.2f\t%.2f\t%.2f\n' % (probe_acc[0], probe_acc[1], probe_acc[2]))
    print(predict_path)
    print(probe_acc)

def market_eval(source, transform_dir):
    target = 'market'
    test_pair_predict(source + '_model.h5',
                           transform_dir + '/query', transform_dir + '/bounding_box_test',	
                          source + '_' + target + '_pid.log', source + '_' + target + '_score.log')


if __name__ == '__main__':
    
    market_result_eval('cross_filter_pid.log')
