
import os
import re
import random
import sys
import csv
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, model_from_json
import keras.backend as K
import pickle

def create_sent_morph(w_m):
    word, morph = w_m.split("_")[0], w_m.split("_")[1]
    if morph in ["助詞","助動詞","記号","接続詞"]:
        res = w_m
    else:
        res = "<{}>".format(morph)
    return res

def choise_output_word_id(distribution, mode='greedy'):
    output_ids = np.argsort(distribution)[::-1]
    def check(id):
        if id == 0:
            return False
        elif id_to_word[id] == "<bos>":
            return False
        return True

    if mode == "greedy":
        i = 0
        while True:
            output_id = output_ids[0]
            if check(output_id):
                break
            else:
                i += 1
    elif mode == "random":
        output_ids = output_ids[:3]
        while True:
            output_id = random.choice(output_ids)
            if check(output_id):
                break
    else:
        raise ValueError("modeの値が間違っています")

    return output_id

def touch_file(file_path):
    with open(file_path,"w") as fo:
        pass
    return

if __name__ == '__main__':
    """
    必要なファイルは
    1. ロードするモデルのjson
    2. 重みファイルhdf5
    3. word_to_idが入ったpickle
    """
    # model_dir = "./templete_model/models_5000"
    model_dir = "./language_model/models_wiki_edojidai"
    model_fname = "{}/model.json".format(model_dir)
    weights_dir = model_dir+"/weights"
    weights = "weights.hdf5"
    weights_fname = "{}/{}".format(weights_dir,weights)
    word2id_fname = "{}/word2id.p".format(model_dir)
    save_sample_fname = "{}/sample_result_{}.csv".format(model_dir,weights)
    touch_file(save_sample_fname)

    with open(word2id_fname,"rb") as fi:
        word_to_id = pickle.load(fi)
    id_to_word = {i:w for w,i in word_to_id.items()}

    with open(model_fname,"r") as fi:
        model_json = fi.read()
    model = model_from_json(model_json)
    model.load_weights(weights_fname)
    # model.summary()
    # import pdb; pdb.set_trace()
    n_samples = 50
    maxlen = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    h_length = model.layers[2].get_output_at(0).get_shape().as_list()[1]
    print('----- Generating text -----')
    for n_sample in range(n_samples):
        x_pred = np.zeros(shape=(1,maxlen),dtype='int32')
        x_pred[0,0] = word_to_id['<bos>']
        pred_h = np.random.normal(0,1,(1,h_length))
        pred_c = np.random.normal(0,1,(1,h_length))
        sentence = ["<bos>"]
        for i in range(maxlen-1):
            preds = model.predict([x_pred,pred_h,pred_c], verbose=0)[0]
            output_id = choise_output_word_id(preds[i],mode='random')
            output_word = id_to_word[output_id]
            sentence.append(output_word)
            if output_word == "<eos>":
                break
            x_pred[0,i+1] = output_id
        if "<eos>" not in sentence:
            err_mes = "<eos> did not appear"
            print(err_mes)
            continue
    
        # <bos>と<eos>を削除
        del sentence[0]
        del sentence[-1]
    
        sent_surface = [w_m.split("_")[0] for w_m in sentence]
        sent_surface = " ".join(sent_surface)
        sent_morph = [create_sent_morph(w_m) for w_m in sentence]
    
        print(sent_surface)
        with open(save_sample_fname,"a") as fo:
            writer = csv.writer(fo)
            writer.writerow([sent_surface, " ".join(sent_morph)])
            

