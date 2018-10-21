
import os
import re
import random
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, model_from_json
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
import pickle

from utils import sent_to_surface_conjugated, max_sent_len, save_config
from utils import create_words_set, create_data_x_y, create_emb_and_dump
from utils import plot_history_loss, choise_output_word_id, add_funcword
from utils import lstm_predict_sampling

def on_epoch_end(epoch, logs):

    BorEOS = "<BOS/EOS>_BOS/EOS".lower()
    print('----- Generating text after Epoch: %d' % epoch)
    sent_surface, sent_morph = lstm_predict_sampling(
                                    model=model,
                                    maxlen=maxlen,
                                    word_to_id=word_to_id,
                                    id_to_word=id_to_word,
                                    h_length=h_length,
                                    is_reversed=is_reversed)
    print(sent_surface)
    return

def data_check():
    # コピーのソースの確認
    if not os.path.exists(data_fname):
        raise IOError(f"{data_fname}がありません")
    # 各モデルを保存するベースDirの確認
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    # モデルのDirの確認
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # 重みを保存するDirの確認
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    if is_lang_model:
        if not os.path.exists(func_wordsets_fname):
            raise IOError(f"{func_wordsets_fname}がありません")
    if is_data_analyzed:
        if not os.path.exists(save_data_fname):
            raise IOError("保存されたanalyzed dataがありません")

    return

def create_model(save_path:str,
                 h_length=32):
    main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
    emb = Embedding(input_dim=words_num+1,
                    output_dim=w2v_dim,
                    input_length=maxlen,
                    weights=[embedding_matrix],
                    mask_zero=True,
                    trainable=True
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

    model_json = model.to_json()
    with open(save_path, mode='w') as fo:
        fo.write(model_json)

    return model

if __name__ == '__main__':

    data_fname = "./source/copy_temp.txt"
    # data_fname = "./source/wiki_edojidai.txt"
    # base_dir = "templete_model"
    base_dir = "templete_model"
    # model_dir_name = "models_5000"
    model_dir_name = "model_temp"
    func_wordsets_fname = "func_wordsets.p"
    w2v_fname = "model.bin"
    maxlen = 40
    mecab_lv = 2
    save_weight_period = 20
    epochs = 10
    batch_size = 32
    is_data_analyzed = False
    is_lang_model = False
    is_reversed = False
    use_loaded_emb = False
    use_loaded_model = False
    use_loaded_weight = False
    use_conjugated = False

    model_dir = os.path.join(base_dir,model_dir_name)
    weights_dir = os.path.join(model_dir,"weights")
    w2v_emb_fname = os.path.join(model_dir, "emb_wordsets.p")
    save_w2i_fname = os.path.join(model_dir, "word2id.p")
    save_data_fname = os.path.join(model_dir, "analyzed_data.txt")
    save_model_fname = os.path.join(model_dir, "model.json")
    save_weights_fname = os.path.join(weights_dir, "weights.hdf5")
    save_callback_weights_fname = os.path.join(weights_dir,"weights_{epoch:03d}_{loss:.2f}.hdf5")
    save_config_fname = os.path.join(model_dir, "config.txt")
    save_loss_fname = os.path.join(model_dir, "loss.png")
    data_check()

    # 解析済みデータをロードするならここは必要ない
    if is_data_analyzed:
        with open(save_data_fname,"r") as fi:
            sent_list = fi.readlines()
        print("解析済みデータをロードしました")
    else:
        print("データを解析します")
        with open(data_fname, "r") as fi:
            data = fi.read()
        sent_list = sent_to_surface_conjugated(
                        data,
                        save_path=save_data_fname,
                        level=mecab_lv,
                        use_conjugated=use_conjugated)
    sent_list = [sent.strip() for sent in sent_list if 3 <= len(sent.split(" ")) <= maxlen]

    # 各種データの情報
    n_samples = len(sent_list)
    maxlen = max_sent_len(sent_list) + 1
    words_set = create_words_set(sent_list)
    if is_lang_model:
        print("機能語のセットをロードします")
        with open(func_wordsets_fname,"rb") as fi:
            funcwords_set = pickle.load(fi)
        words_set = sorted(words_set | funcwords_set)
    words_num = len(words_set)

    print("入出力データを作成中")
    X, Y, word_to_id = create_data_x_y(
                                        sent_list,
                                        maxlen,
                                        words_num,
                                        is_reversed=is_reversed)
    if is_lang_model:
        word_to_id = add_funcword(word_to_id, funcwords_set)
        if words_num != len(word_to_id):
            print(words_num, len(word_to_id))
            raise AssertionError("words_numとword_to_idの総数が異なります")

    if use_loaded_emb:
        print("保存されたembをロードします")
        if not os.path.exists(path=w2v_emb_fname):
            raise IOError("w2vのファイルがありません")
        with open(w2v_emb_fname, "rb") as fi:
            embedding_matrix, loaded_words_set = pickle.load(fi)
        if words_set != loaded_words_set:
            raise ValueError("words_setに含まれる単語が一致しません")
    else:
        print("embを作成し、保存します")
        embedding_matrix = create_emb_and_dump(w2v_fname,
                                               words_set,
                                               word_to_id,
                                               w2v_emb_fname)

    w2v_dim = len(embedding_matrix[1])
    id_to_word = {i:w for w,i in word_to_id.items()}
    if use_loaded_model:
        print("保存されたモデルをロードします")
        with open(save_model_fname,"r") as fi:
            json_string = fi.read()
            model = model_from_json(json_string)
    else:
        print("モデルを構築します")
        model = create_model(save_model_fname)

    if use_loaded_weight:
        print("重みをロードしました")
        model.load_weights(save_weights_fname)

    model.compile(optimizer='adam', loss="categorical_crossentropy")
    model.summary()
    h_length = model.layers[2].get_output_at(0).get_shape().as_list()[1]
    save_dict = {"n_samples":str(n_samples),
                 "maxlen":str(maxlen),
                 "words_num":str(words_num),
                 "h_length":str(h_length),
                 "w2v_dim":str(w2v_dim),
                 "is_reversed":str(is_reversed),
                 "mecab_lv":str(mecab_lv),
                 "use_conjugated":str(use_conjugated)
                 }
    save_config(path=save_config_fname, save_dict=save_dict)
    with open(save_w2i_fname, "wb") as fo:
        pickle.dump([word_to_id, is_reversed], fo)
    """
        callbacksの記述
    """
    # es_cb = keras.callbacks.EarlyStopping(patience=30,
    #                                       verbose=1,
    #                                       mode='auto')
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model_checkpoint = ModelCheckpoint(filepath=save_callback_weights_fname,
                                        save_weights_only=True,
                                        period=save_weight_period)
    loss_history = []
    val_loss_history = []
    for i in range(epochs):
        h = np.random.normal(0,1,(n_samples,h_length))
        c = np.random.normal(0,1,(n_samples,h_length))
        fit = model.fit(x=[X, h, c],
                y=Y,
                epochs=i+1,
                batch_size=batch_size,
                initial_epoch = i,
                validation_split=0.05,
                callbacks=[print_callback,model_checkpoint])
        loss = fit.history['loss'][0]
        val_loss = fit.history['val_loss'][0]
        loss_history.append(loss)
        val_loss_history.append(val_loss)
    model.save_weights(save_weights_fname)
    plot_history_loss(loss_history, val_loss_history, save_loss_fname)
