# -*- coding: utf-8 -*-
"""
@time: 8/28/24 2:12 PM
word-wise
"""

import logging

import pandas as pd
from tqdm import tqdm

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from  scipy.stats import pearsonr, spearmanr


from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch


def reg_with_hc(feats, fpth, vpth):
    """regression with hand-crafted features"""
    logging.info("feature: {}".format(feats))

    # load data
    df = pd.read_csv(fpth, sep="\t")
    df = df.astype({"id": int, "index": int})

    print("sample size:", df.shape[0])
    ref = pd.read_csv(vpth, sep="\t")

    # add feat info
    for feat in feats:
        df[feat] = [ref[ref["word"] == w][feat].item() for w in df["word"]]

    # train, val, test split
    train, val, test = df[df["set"] == "train"], df[df["set"] == "val"], df[df["set"] == "test"]
    print("train size: ", train.shape)
    print("test size: ", test.shape)
    print("val size: ", val.shape)

    train_x, train_y = train[feats].to_numpy(), train["complexity"].to_numpy().reshape((-1, 1))
    test_x, test_y = test[feats].to_numpy(), test["complexity"].to_numpy().reshape((-1, 1))
    val_x, val_y = val[feats].to_numpy(), val["complexity"].to_numpy().reshape((-1, 1))
    #
    reg = Ridge()
    reg.fit(train_x, train_y)

    # eval, MAE, R2, and Spearmanr
    test_p = reg.predict(test_x)

    print("test: ")
    evaler(test_y, test_p)


def evaler(gt, pred):
    """evaluation"""
    print("MAE: ", mean_absolute_error(gt, pred))
    print("R2: ", r2_score(gt, pred))
    print("Pearson:  ", pearsonr(gt.flatten(), pred.flatten()))
    print("Spearmanr: ",  spearmanr(gt.flatten(), pred.flatten()))


def reg_with_emb(fpth, feats):

    # load data
    df = pd.read_csv(fpth, sep="\t")
    df = df.astype({"id": int, "index": int})
    logging.info("sample size: {}".format(df.shape[0]))
    print(df.columns.tolist())
    df = pd.read_json(epth) if epth.exists() else prepare_embs(df, epth)

    # + hc?
    if feats:
        ref = pd.read_csv(vpth, sep="\t")
        for feat in feats:
            # add feat info
            df[feat] = [ref[ref["word"] == w][feat].item() for w in df["word"]]
    print(df.columns.tolist())

    # train, val, test split
    train, val, test = df[df["set"]=="train"], df[df["set"]=="val"], df[df["set"]=="test"]
    print("train: ", train.shape)
    print("test: ", test.shape)
    print("val: ", val.shape)

    train_x, train_y = train["emb"], train["complexity"].to_numpy().reshape((-1, 1))
    train_x = np.array([np.array(i).squeeze() for i in train_x])
    test_x, test_y = test["emb"], test["complexity"].to_numpy().reshape((-1, 1))
    test_x = np.array([np.array(i).squeeze() for i in test_x])
    val_x, val_y = val["emb"], val["complexity"].to_numpy().reshape((-1, 1))
    val_x = np.array([np.array(i).squeeze() for i in val_x])

    if feats:
        train_x = np.concatenate([train_x, train[feats].to_numpy()], axis=1)
        test_x = np.concatenate([test_x, test[feats].to_numpy()], axis=1)
        val_x = np.concatenate([val_x, val[feats].to_numpy()], axis=1)

    reg = Ridge()
    reg.fit(train_x, train_y)

   
    test_p = reg.predict(test_x)
    print("test: ")
    evaler(test_y, test_p)


def prepare_embs(df, epth):
    device = "cuda" 
    SYM = "[MASK]"
    model_name = "hfl/cino-large"  # argue v2 doesnt support eng

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, max_length=512)
    tokenizer.add_tokens(new_tokens=SYM, special_tokens=SYM)
    model = XLMRobertaModel.from_pretrained(model_name)
    model.to(device)
    SYM_ID = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(SYM))[0]
    SIGN = tokenizer.convert_ids_to_tokens([6])[0]
    
    df["emb"] = [torch.tensor(0)] * len(df)
    pbar = tqdm(total=len(df))
    for _, row in df.iterrows():
        word = row["word"]
        sent = row["sentence"]
        index = row["index"]

        assert sent[index - 1: index + len(word) - 1] == word

        sent = sent[:index - 1] + SYM + word + SYM + sent[index + len(word) - 1:]
        input_ids = torch.tensor(tokenizer.encode(sent))
        pos = np.where(input_ids == SYM_ID)[0]  # mask position
        assert pos.size == 2
        assert (input_ids[pos] / SYM_ID).mean() == 1

        # revise input_ids & create att mask
        front, mid, end = input_ids[:pos[0]], input_ids[pos[0] + 1: pos[-1]], input_ids[pos[-1]+1:]
        
        input_ids = torch.concat([front, mid, end]) 
    
        pad = torch.tensor([1] * max(0, max_length - len(input_ids)))
        input_ids = torch.concat([input_ids, pad]).unsqueeze(0)

        att_msk = (input_ids != 1) * 1

        word_pos = [i+pos[0] for i in range(len(mid))]
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze()[word_pos])
        # if  "".join(tokens).strip(SIGN) != word:
        #     breakpoint()
        assert (3 not in input_ids.squeeze()[word_pos]) == ("".join(tokens).strip(SIGN) == word)
        
        assert ((att_msk == 1) == (input_ids != 1)).all() 
        assert att_msk.shape[1] == input_ids.shape[1] == max_length

        inputs = {"input_ids": input_ids.to(device), "attention_mask": att_msk.to(device)}

        with torch.no_grad():
            outputs = model(**inputs)
        
        emb = outputs.last_hidden_state.squeeze()[word_pos].mean(dim=0)
        df.at[_, "emb"] = emb.unsqueeze(dim=0).cpu().numpy()
        pbar.update(1)
    pbar.close()
    df.to_json(epth)
    logging.info("word embs save to {}".format(epth.as_posix()))
    return df


if __name__ == "__main__":

    fpth = ""  # path to rating file
    vpth = ""  # path to vocab info file
    epth = ""  # path to file with embs

    """with hc feats"""
    # regression with one hc feat
    feats = ["strokes", "log freq", "word length"]
    for feat in feats:
        reg_with_hc([feat], fpth, vpth) 

    """reg with emb"""
    # feats=["log freq"]
    # reg_with_emb(epth, feats)











