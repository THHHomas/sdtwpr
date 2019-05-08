import torch
import torch as t
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt

from utils.file_helper import write, safe_remove

import numpy as np

from pair_train import load_and_process, input_shape
avgpool = nn.AdaptiveAvgPool2d((1, 1))
device=t.device("cuda")

def test_pair_predict(pair_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    # todo
    model = torch.load(pair_model_path).eval()
    test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)
    '''y_pred=Variable(y_pred, requires_grad=True)

    print("before  load_model:----------------------------------------------------------------- ")
    model = load_model("./market_pair_pretrain.h5", custom_objects={'triplet_hard_loss': triplet_hard_loss})
    print("model finished:----------------------------------------------------------------- ")
    # model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    #model = Model(inputs=[model.layers[0].get_input_at(0)],
    #              outputs=[model.get_layer('ave_pooled_part').get_output_at(0)])
    for layer in model.layers:
        print("!!!!:",layer)
    print("@@@@@",model.layers[-5])
    #model = Model(inputs=[model.layers[0].get_input_at(0)],
    #              outputs=[model.layers[-4].get_output_at(0)] )

    global_feature = model.layers[-5].get_output_at(0)
    feature1 = GlobalAveragePooling2D(name='ave_pooled')(global_feature)

    model = Model(inputs=[model.input], outputs=[feature1])
    # model = Model(inputs=[model.input], outputs=[model.get_layer('avg_pool').output])
    print("before  test_predict:----------------------------------------------------------------- ")
    test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)'''





def test_predict(net, probe_path, gallery_path, pid_path, score_path):
    #net = Model(inputs=[net.input], outputs=[net.output])
    #print("output shape:", net.output.shape)
    test_f, test_info = extract_feature(gallery_path, net)  ###19732
    query_f, query_info = extract_feature(probe_path, net)  ##3368
    result, result_argsort = sort_similarity(query_f, test_f)

    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    safe_remove(pid_path)
    safe_remove(score_path)
    np.savetxt(pid_path, result_argsort, fmt='%d')
    np.savetxt(score_path, result, fmt='%.4f')



def get_augmentation_batch(image):
    #Resize it correctly, as needed by the test time augmentation.
    #im_mean = np.asarray([103.0626238, 115.90288257, 123.15163084], dtype=np.float32)
    #image = cv2.resize(image, (224+32, 224+32))

    #Change into CHW format
    #image = np.rollaxis(image,2)
    image=np.squeeze(image)
    #Setup storage for the batch
    #print("image shape:", image.shape)
    batch = np.zeros((2, input_shape[0], input_shape[1], 3), dtype=np.float32)

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
        x = load_and_process(image_path)
        x = np.expand_dims(x, axis=0)
        x = get_augmentation_batch(x)
        
        '''        
        cv2.imshow("pre", x[0,:,:,:])
        cv2.waitKey(1000)
        cv2.imshow("pre", x[1,:,:,:])
        cv2.waitKey(1000)
        '''

        x= t.Tensor(np.transpose(x,(0,3,1,2))).to("cuda")
        feature = net(x)
        #y_ =t.sqrt(t.sum(feature**2, 1)).unsqueeze(1)
        #feature = feature/y_
        
        feature = avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = feature.cpu().detach().numpy()
        #print(feature.shape)
        #feature[0]=np.squeeze(feature[0])
        #print(feature[0].shape, feature[1].shape)
        #print(len(feature), type(feature))
        #print(feature[0].shape)
        #feature = np.concatenate(feature,axis = 1)
        #print(feature.shape)
	        

        feature = np.mean(feature, axis=0)
        #print("feature.shpae: ",feature.shape)
        features.append(np.squeeze(feature))
        infos.append((person, camera))
        #print(feature.shape,np.max(feature))
    return features, infos


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

'''
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
'''

def similarity_matrix(query_f, test_f):
    query_f = t.from_numpy(np.array(query_f)).to(device)
    test_f = t.from_numpy(np.array(test_f)).to(device)
    query_f = query_f/t.norm(query_f, dim=1).unsqueeze(1)
    test_f = test_f/t.norm(test_f, dim=1).unsqueeze(1)
    
    result = t.mm(query_f, test_f.t()).cpu().numpy()
    
    # descend
    return result


def sort_similarity(query_f, test_f):
    result = similarity_matrix(query_f, test_f)
    result_argsort = np.argsort(-result, axis=1)
    return result, result_argsort

def market_eval(source, transform_dir):
    target = 'market'
    test_pair_predict(source + '_model.h5',
                           transform_dir + '/query', transform_dir + '/bounding_box_test',	
                          source + '_' + target + '_pid.log', source + '_' + target + '_score.log')


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

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    market_eval('market', '../dataset/Market-1501-v15.09.15')
    market_result_eval('market_market_pid.log',
                            TEST='../dataset/Market-1501-v15.09.15/bounding_box_test',
                            QUERY='../dataset/Market-1501-v15.09.15/query')


