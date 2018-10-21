
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

from utils import choise_output_word_id, lstm_predict_sampling, create_sent_morph

if __name__ == '__main__':
    """
    必要なファイルは
    1. ロードするモデルのjson
    2. 重みファイルhdf5
    3. word_to_idとis_reversedが入ったpickle
    """
    model_dir = "./templete_model/model_temp"
    # model_dir = "./language_model/wiki_edojidai"
    model_fname = f"{model_dir}/model.json"
    weights_dir = model_dir+"/weights"
    weights = "weights.hdf5"
    weights_fname = f"{weights_dir}/{weights}"
    word2id_fname = f"{model_dir}/word2id.p"
    save_sample_fname = f"{model_dir}/sampling_{weights}.csv"

    with open(word2id_fname,"rb") as fi:
        word_to_id, is_reversed = pickle.load(fi)
    id_to_word = {i:w for w,i in word_to_id.items()}

    with open(model_fname,"r") as fi:
        model_json = fi.read()
    model = model_from_json(model_json)
    model.load_weights(weights_fname)

    n_samples = 10
    maxlen = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    h_length = model.layers[2].get_output_at(0).get_shape().as_list()[1]

    print('----- Generating text -----')
    surface_morph = []
    for n_sample in range(n_samples):
        sent_surface, sent_morph = lstm_predict_sampling(
                                        model=model,
                                        maxlen=maxlen,
                                        word_to_id=word_to_id,
                                        id_to_word=id_to_word,
                                        h_length=h_length,
                                        is_reversed=is_reversed)
        print(sent_surface)
        surface_morph.append([sent_surface,sent_morph])
    with open(save_sample_fname,"w") as fo:
        writer = csv.writer(fo)
        writer.writerows(surface_morph)
