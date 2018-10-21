
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
import pickle

from utils import choise_output_word_id

def create_sent_morph(w_m):
    word, morph = w_m.split("_")[0], w_m.split("_")[1]
    if morph in ["助詞","助動詞","記号","接続詞"]:
        res = w_m
    else:
        m = w_m.split("_")[1:]
        m = "_".join(m)
        res = "<{}>".format(m)
    return res

if __name__ == '__main__':
    """
    必要なファイルは
    1. ロードするモデルのjson
    2. 重みファイルhdf5
    3. word_to_idが入ったpickle
    """
    # model_dir = "./templete_model/models_5000"
    model_dir = "./language_model/wiki_edojidai"
    model_fname = f"{model_dir}/model.json"
    weights_dir = model_dir+"/weights"
    weights = "weights.hdf5"
    weights_fname = f"{weights_dir}/{weights}"
    word2id_fname = f"{model_dir}/word2id.p"
    save_sample_fname = f"{model_dir}/sampling_{weights}.csv"

    with open(save_sample_fname, "w") as fo:
        pass
    with open(word2id_fname,"rb") as fi:
        word_to_id, is_reversed = pickle.load(fi)
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
    BorEOS = "<BOS/EOS>_BOS/EOS_*_*_*_*".lower()

    print('----- Generating text -----')
    for n_sample in range(n_samples):
        x_pred = np.zeros(shape=(1,maxlen),dtype='int32')
        x_pred[0,0] = word_to_id[BorEOS]
        h_pred = np.random.normal(0,3,(1,h_length))
        c_pred = np.random.normal(0,3,(1,h_length))
        sentence = []
        for i in range(maxlen-1):
            preds = model.predict([x_pred,h_pred,c_pred], verbose=0)[0]
            output_id = choise_output_word_id(preds[i], mode="greedy")
            output_word = id_to_word[output_id]
            sentence.append(output_word)
            if output_word == BorEOS:
                break
            x_pred[0,i+1] = output_id
        if sentence[-1] != BorEOS:
            err_mes = "produce_failed!"
            print(err_mes)
        elif not sentence:
            err_mes = "white"
            print(err_mes)

        del sentence[-1]
        sent_surface = [w_m.split("_")[0] for w_m in sentence]
        if is_reversed:
            sent_surface = [word for word in reversed(sent_surface)]
        sent_surface = " ".join(sent_surface)
        print(sent_surface)
    
        sent_morph = [create_sent_morph(w_m) for w_m in sentence]    
        with open(save_sample_fname,"a") as fo:
            writer = csv.writer(fo)
            writer.writerow([sent_surface, " ".join(sent_morph)])
            

