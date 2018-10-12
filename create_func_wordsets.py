
import pickle

def morph_check(w_m:str):
    func_set = ["助詞","助動詞","記号","接続詞"]
    if w_m in ["<bos>", "<eos>"]:
        return False
    if w_m.split("_")[1] in func_set:
        return True
    else:
        return False
        

if __name__ == '__main__':
    emb_wordsset_fname = "emb_wordsets.p"
    save_fname = "func_wordsets.p"
    with open(emb_wordsset_fname,"rb") as fi:
        emb, wordsets = pickle.load(fi)
    
    func_words_sets = {w_m for w_m in wordsets if morph_check(w_m)}

    print(func_words_sets)
    with open(save_fname, "wb") as fo:
        pickle.dump(func_words_sets, fo)