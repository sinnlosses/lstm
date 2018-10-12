
import os
import re
import random
import sys
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers.noise import GaussianNoise
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from gensim.models import KeyedVectors
from keras.callbacks import LambdaCallback, ModelCheckpoint
import keras.backend as K
import pickle
import MeCab
import tensorflow as tf
import matplotlib.pyplot as plt

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
        result = "{} {} {}".format("<bos>",
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
    w2v_fname = "jawiki_word.bin"
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

def plot_history_loss(loss_history):
    # Plot the loss in the history
    fig = plt.figure(1)
    plt.plot(loss_history,label="loss for training")
    plt.title("model_loss")
    plt.xlabel("epoch")
    plt.ylabel('loss: cross_entropy')
    plt.legend(loc='upper right')
    fig.savefig(save_loss_fname)

    return

if __name__ == '__main__':

    data_fname = "./source/copy_source.txt"
    base_dir = "templete_model"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    model_dir = "{}/models_5000".format(base_dir)
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
    X = [sent[:-1] for sent in sent_seq]
    X = sequence.pad_sequences(sequences=X,
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
    h_length = 32

    main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
    emb = Embedding(input_dim=words_num+1,
                    output_dim=w2v_dim,
                    input_length=maxlen,
                    weights=[embedding_matrix],
                    mask_zero=True,
                    trainable=False
                    )(main_input)
    initial_h = Input(shape=(h_length,),
                          dtype='float32',
                          name='initial_h')
    initial_c = Input(shape=(h_length,),
                          dtype='float32',
                          name='initial_c')
 
    lstm_out = LSTM(h_length,
                    return_sequences=True)(emb,
                                initial_state=[initial_h,
                                               initial_c])
    main_output = TimeDistributed(
                    Dense(words_num+1,
                        activation='softmax',
                        name='main_output'))(lstm_out)
 
    model = Model(inputs=[main_input, initial_h,initial_c],
                    outputs=main_output)
    loss = keras.losses.categorical_crossentropy
    model.compile(optimizer='rmsprop', loss=loss)
    model.summary()
    """
        callbacksの記述
    """
    es_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=0, verbose=1, mode='auto')
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model_checkpoint = ModelCheckpoint(filepath=save_callback_weights_fname,
                                        monitor='loss',
                                        save_weights_only=True,
                                        period=10)
    epochs = 150
    loss_history = []
    for i in range(epochs):
        h = np.random.normal(0,1,(n_samples,h_length))
        c = np.random.normal(0,1,(n_samples,h_length))
        fit = model.fit(x=[X, h, c],
                y=Y,
                epochs=i+1,
                batch_size=32,
                initial_epoch = i,
                callbacks=[print_callback,es_cb,model_checkpoint])
        loss = fit.history['loss'][0]
        loss_history.append(loss)
    model_json = model.to_json()
    with open(save_model_fname, mode='w') as fo:
        fo.write(model_json)
    model.save_weights(save_weights_fname)
    plot_history_loss(loss_history)
    save_config()
    with open(save_word2id_fname,"wb") as fo:
        pickle.dump(word_to_id, fo)