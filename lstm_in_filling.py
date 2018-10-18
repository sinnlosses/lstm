
import re
import random
import sys
import os
import numpy as np
from keras.layers import Input, Dense, Embedding, LSTM
from keras.models import Model, model_from_json
from keras.layers.wrappers import TimeDistributed
from gensim.models import KeyedVectors
from keras.callbacks import LambdaCallback
import keras.backend as K
import pickle
import csv

def touch_file(file_path):
    with open(file_path,"w") as fo:
        pass
    return

def choise_output_word_id(distribution, morph, mode='greedy'):
    output_ids = np.argsort(distribution)[::-1]
    morph = morph.strip("<>")
    def check(id):
        if id == 0:
            return False
        elif id_to_word[id] in ["<bos>", "<eos>"]:
            return False
        elif id_to_word[id].split("_")[1] != morph:
            return False
        return True

    if mode == "greedy":
        i = 0
        while True:
            output_id = output_ids[i]
            if check(output_id):
                break
            else:
                i += 1
    elif mode == "random":
        output_ids = output_ids[i]
        choices = []
        while True:
            output_id = output_ids[i]
            if check(output_id):
                choices.append[output_id]
                if len(choices) > 2:
                    break
            else:
                i += 1
        output_id = random.choice(choices)
    else:
        raise ValueError("modeの値が間違っています")

    return output_id

def create_sava_dir(temp_dir, lang_dir):
    temp_dir_last_name = temp_dir.split("/")[-1]
    lang_dir_last_name = lang_dir.split("/")[-1]

    save_dir = "./{}_{}".format(temp_dir_last_name, lang_dir_last_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    return save_dir


if __name__ == '__main__':

    templete_dir = "./templete_model/models_5000_fast_isTraining"
    templete_csv_fname = "{}/sample_result_weights.hdf5_0_3.csv".format(templete_dir)
    # templete_csv_fname = "{}/temp.csv".format(templete_dir)   
    lang_dir = "./language_model/models_wiki_edojidai"
    lang_model_fname = "{}/model.json".format(lang_dir)
    lang_weights_fname = "{}/weights/weights.hdf5".format(lang_dir)
    save_dir = create_sava_dir(templete_dir,lang_dir)
    save_fname = "{}/in_filling_result.txt".format(save_dir)
    word2id_fname = "{}/word2id.p".format(lang_dir)

    touch_file(save_fname)
    with open(templete_csv_fname,"r") as fi:
        templete = csv.reader(fi)
        templete = [t for t in templete]

    with open(lang_model_fname,"r") as fi:
        model = model_from_json(fi.read())
    model.load_weights(lang_weights_fname)

    maxlen = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    h_length = model.layers[2].get_output_at(0).get_shape().as_list()[1]

    with open(word2id_fname, "rb") as fi:
        word_to_id = pickle.load(fi)
    id_to_word = {i:w for w,i in word_to_id.items()}

    n_samples = 30
    if len(templete) < n_samples:
        raise ValueError("サンプル数が上限を超えています\ntemplete: ".format(str(len(templete))))
 
    samples = random.sample(templete,n_samples)
    for sample in samples:
        sample_1 = sample[1]
        sample_wakati_list = sample_1.split(" ")
        x_pred = np.zeros(shape=(1,maxlen),dtype='int32')
        x_pred[0,0] = word_to_id['<bos>']
        pred_h = np.random.normal(0,3,(1,h_length))
        pred_c = np.random.normal(0,3,(1,h_length))
        sentence = []

        for i in range(len(sample_wakati_list)):
            if not re.match(r"<.+>",sample_wakati_list[i]):
                if i < maxlen-1:
                    x_pred[0,i+1] = word_to_id[sample_wakati_list[i]]
                sentence.append(sample_wakati_list[i].split("_")[0])
                continue

            preds = model.predict([x_pred,pred_h,pred_c], verbose=0)[0]
            output_id = choise_output_word_id(preds[i], morph=sample_wakati_list[i],mode="greedy")
            word = id_to_word[output_id]
            if i < maxlen-1:
                x_pred[0,i+1] = output_id
            sentence.append(word.split("_")[0])

        # print(sample)
        print(" ".join(sentence))
        with open(save_fname,"a") as fo:
            fo.write("-----\n")
            fo.write(sample[0]+"\n")
            fo.write(sample[1]+"\n")
            fo.write(" ".join(sentence)+"\n")
            fo.write("-----\n")
