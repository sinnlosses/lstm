
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import random
import sys

import keras
from keras.layers import Lambda, Input, Dense, Embedding, LSTM, Bidirectional
from keras.layers import RepeatVector, Layer, Activation, Dropout
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.layers.noise import GaussianNoise
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import LambdaCallback, ModelCheckpoint
import keras.backend as K
import tensorflow as tf

import numpy as np
import matplotlib as plt
from gensim.models import KeyedVectors
import pickle
import MeCab
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

def on_epoch_end(epoch, logs):
 
    print('----- Generating text after Epoch: %d' % epoch)
    x_pred = np.zeros(shape=(1,maxlen),dtype='int32')
    x_pred[0,0] = word_to_id['<bos>']
    pred_h = np.random.normal(0,1,(1,h_length))
    pred_c = np.random.normal(0,1,(1,h_length))
    sentence = ["<bos>"]
    for i in range(maxlen-1):
        preds = model.predict([x_pred,pred_h,pred_c], verbose=0)[0]
        output_id = choise_output_word_id(preds[i], mode="random")
        output_word = id_to_word[output_id]
        sentence.append(output_word)
        if output_word == "<eos>":
            break
        x_pred[0,i+1] = output_id
    if "<eos>" not in sentence:
        err_mes = "produce_failed!\n"
        print(err_mes, end="")
        with open(save_gen_morph_fname,"a") as fo:
            fo.write(err_mes)
        return
 
    # <bos>と<eos>を削除
    del sentence[0]
    del sentence[-1]
 
    def create_sent_morph(w_m):
        word, morph = w_m.split("_")[0], w_m.split("_")[1]
        if morph in ["助詞","助動詞","記号","接続詞"]:
            res = word
        else:
            res = morph
        return res
 
    sent_surface = [w_m.split("_")[0] for w_m in sentence]
    sent_surface = " ".join(sent_surface)
    sent_morph = [create_sent_morph(w_m) for w_m in sentence]
 
    print(sent_surface)
 
    with open(save_gen_morph_fname,"a") as fo:
        res = "{},{}\n".format(sent_surface, sent_morph)
        fo.write(res)
    return
 
def max_sent_len(sent_list):
    return max([len(sent.split(" ")) for sent in sent_list])
 
def sent_to_surface_conjugated(sentences:str):
    """
        \nで区切られた文章群の各単語mecabで解析し表層形_活用形の形式で保存する
        また、表層形_活用形の形式で文章群を書き直し保存する
    """
    mecab = MeCab.Tagger()
    mecab.parse("")
    sent_list = sentences.split("\n")
    mecabed_list = []
    for sent in sent_list:
        word_morph_list = []
        node = mecab.parseToNode(sent)
        while node:
            word = node.surface
            morph = node.feature.split(",")[0]
            if morph == "BOS/EOS":
                node = node.next
                continue
            wm = "{}_{}".format(word, morph)
            word_morph_list.append(wm)
            node = node.next
        result = "{} {} {}".format("<eos>",
                                   " ".join(word_morph_list),
                                   "<eos>")
        mecabed_list.append(result)
    return mecabed_list
 
def create_emb_and_dump(words_set, word_to_id,fname="emb_wordsets.p"):
    """
    embのファイルパスを指定すればembmatrixをロード、
    指定しなければembmatrixを作成して
    """
    if os.path.exists(path=fname):
        with open(fname, "rb") as fi:
            emb, loaded_words_set = pickle.load(fi)
        if words_set == loaded_words_set:
            return emb
        else:
            raise ValueError("words_setに含まれる単語が一致しません")
 
    print("w2vデータをload中...")
    w2v_fname = "model.bin"
    words_num = len(words_set)
    w2v = KeyedVectors.load_word2vec_format(w2v_fname, binary=True)
    w2v_dim = w2v.vector_size
    embedding_matrix = np.zeros((words_num+1, w2v_dim))
    for word in words_set:
        word_surface = word.split("_")[0]
        id = word_to_id[word]
        if word_surface not in w2v.vocab:
            embedding_matrix[id] = np.random.normal(0,1,(1,w2v_dim))
        else:
            embedding_matrix[id] = w2v.wv[word_surface]
    with open(fname,"wb") as fo:
        pickle.dump([embedding_matrix, words_set],fo)
 
    return embedding_matrix
 
def touch_file(file_path):
    with open(file_path,"w") as fo:
        pass
    return

def save_config():
    """
        sava_config
    """
    with open(save_config_fname,"w") as fo:
        fo.write("Config\n\n")
        fo.write("maxlen: {}\n".format(maxlen))
        fo.write("n_samples: {}\n".format(n_samples))
        fo.write("words: {}\n".format(words_num))
        fo.write("h_length: {}\n".format(h_length))
        fo.write("w2v_dim: {}\n".format(w2v_dim))
        fo.write("words_example:\n\n")
        for i in words_set[:30]:
            fo.write("{}, ".format(i))
    return

def plot_history_loss(loss_history,val_loss_history):
    # Plot the loss in the history
    fig = plt.figure(1)
    plt.plot(loss_history,label="loss for training")
    plt.plot(val_loss_history, label="val_loss for training")
    plt.title("model_loss")
    plt.xlabel("epoch")
    plt.ylabel('loss: cross_entropy')
    plt.legend(["loss","val_loss"],loc='upper right')
    fig.savefig(save_loss_fname)

    return

if __name__ == '__main__':

    data_fname = "./source/copy_source.txt"
    base_dir = "templete_model"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    model_dir = "{}/models_5000_vae".format(base_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    weights_dir = model_dir+"/weights"
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    w2v_emb_fname = "./{}/emb_wordsets.p".format(model_dir)
    save_model_fname = "./{}/model.json".format(model_dir)
    save_weights_fname = "./{}/weights/weights.hdf5".format(model_dir)
    save_callback_weights_fname = "./"+model_dir+"/weights/weights_{epoch:03d}-{loss:.2f}.hdf5"
    save_gen_morph_fname = "./{}/result.csv".format(model_dir)
    save_config_fname = "./{}/config.txt".format(model_dir)
    save_loss_fname = "./{}/loss.png".format(model_dir)
    save_word2id_fname = "./{}/word2id.p".format(model_dir)

    touch_file(save_gen_morph_fname)
 
    with open(data_fname, "r") as fi:
        data = fi.read()
     
    maxlen = 40
    sent_list = sent_to_surface_conjugated(data)
    sent_list = [sent for sent in sent_list if len(sent.split(" ")) < maxlen]
    n_samples = len(sent_list)
    maxlen = max_sent_len(sent_list) + 10
    sentences = " ".join(sent_list)
 
    words_sequence = text_to_word_sequence(text=sentences,
                                      filters='\n',
                                      split=" ")
    words_set = sorted(set(words_sequence))
    words_num = len(words_set)
 
    tokenizer = Tokenizer(filters='\n')
    tokenizer.fit_on_texts(sent_list)
    sent_seq = tokenizer.texts_to_sequences(sent_list)
    word_to_id = tokenizer.word_index
    id_to_word = tokenizer.index_word
    Enc_X = [sent[1:-1] for sent in sent_seq]
    Enc_X = sequence.pad_sequences(sequences=Enc_X,
                               maxlen=maxlen,
                               padding='post')
    Dec_X = [sent[:-1] for sent in sent_seq]
    Dec_X = sequence.pad_sequences(sequences=Dec_X,
                               maxlen=maxlen,
                               padding='post')
    y_seq = [sent[1:] for sent in sent_seq]
    y_seq = sequence.pad_sequences(sequences=y_seq,
                                   maxlen=maxlen,
                                   padding='post')
    Y = np.zeros((n_samples, maxlen, words_num+1), dtype=np.bool)
 
    for i, id_seq in enumerate(y_seq):
        for j, id in enumerate(id_seq):
            if id == 0:
                continue
            Y[i,j,id] = 1.
    embedding_matrix = create_emb_and_dump(words_set,
                                           word_to_id,
                                           fname=w2v_emb_fname)
    w2v_dim = len(embedding_matrix[1])

    # network parameters
    batch_size = 32
    input_shape = (maxlen,)
    intermediate_dim = 96
    latent_dim = 32
    act = ELU()

    # VAEモデルはEnc-Dec
    # Encoder
    main_input = Input(shape=input_shape, dtype='int32', name='main_input')
    emb = Embedding(input_dim=words_num+1,
                    output_dim=w2v_dim,
                    input_length=maxlen,
                    weights=[embedding_matrix],
                    mask_zero=True,
                    trainable=True
                    )(main_input)
    h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(emb)
    h = Dropout(0.2)(h)
    h = Dense(intermediate_dim, activation='linear')(h)
    h = act(h)
    h = Dropout(0.2)(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    repeated_context = RepeatVector(maxlen)
    decoder_h = LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
    decoder_mean = TimeDistributed(Dense(words_num, activation='linear'))
    h_decoded = decoder_h(repeated_context(z))
    x_decoded_mean = decoder_mean(h_decoded)

    # placeholder loss
    def zero_loss(y_true, y_pred):
        return K.zeros_like(y_pred)

    # Custom VAE loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)
            self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)

        def vae_loss(self, x, x_decoded_mean):
            #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
            labels = tf.cast(x, tf.int32)
            xent_loss = K.sum(tf.contrib.seq2seq.sequence_loss(x_decoded_mean, labels, 
                                                        weights=self.target_weights,
                                                        average_across_timesteps=False,
                                                        average_across_batch=False), axis=-1)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            print(x.shape, x_decoded_mean.shape)
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # we don't use this output, but it has to have the correct shape:
            return K.ones_like(x)

    loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, [loss_layer])
    opt = Adam(lr=0.01) #SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    vae.compile(optimizer='adam', loss=[zero_loss])
    vae.summary()
    
