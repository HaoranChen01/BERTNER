# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:35:28 2020

@author: Haoran6
"""

import pandas as pd
from transformers import BertTokenizer, BertConfig, AdamW
from utils import *
from datasets import BertNerDataset, HundsunDataset
from torch.utils.data import DataLoader, random_split
from models import BertBilstmCRF, BertTcnCRF
from train import *
from collections import defaultdict
import jieba
from functools import reduce
import random


def simple_process():
    """
    小数据集的处理
    """
    train_data = pd.read_excel('data//Train_Data.xlsx')
    text = train_data['text']
    entity = train_data['entity']

    text = [str(t) for t in text]
    entity = [str(et).split(';') for et in entity]

    label = labeling(text[:], entity[:])
    saveList(label, 'data//Train_Label.pkl')


def liu_process():
    """
    刘算法的数据集处理
    """
    raw_contents, raw_labels = raw2sentence('data/liu_data/raw_data.txt')

    split_contents = pd.read_table('data/liu_data/test_data_text.txt', header=None, names=['contents'])['contents'].tolist()
    split_labels = pd.read_table('data/liu_data/test_data_label.txt', header=None, names=['labels'])['labels'].tolist()
    split_labels = [i.split('|')[:-1] for i in split_labels]

    train_contents = raw_contents[:40000]
    train_labels = raw_labels[:40000]
    val_contents = raw_contents[40000:50000]
    val_labels = raw_labels[40000:50000]
    test_contents = raw_contents[50000:] + split_contents
    test_labels = raw_labels[50000:] + split_labels

    tokenizer = BertTokenizer.from_pretrained("albert_chinese_base")
    decoder = DECODER().get('liu_e2t')

    train_encodings = tokenizer(train_contents[::], padding=True, truncation=True, max_length=256, return_tensors="pt")
    length = train_encodings.input_ids.size(1)
    train_labels = LabelProcess(train_labels, length, decoder)
    TrainDataset = BertNerDataset(train_encodings, train_labels)

    val_encodings = tokenizer(val_contents[::], padding=True, truncation=True, max_length=256, return_tensors="pt")
    length = val_encodings.input_ids.size(1)
    val_labels = LabelProcess(val_labels, length, decoder)
    ValDataset = BertNerDataset(val_encodings, val_labels)

    test_encodings = tokenizer(test_contents[::], padding=True, truncation=True, max_length=256, return_tensors="pt")
    length = test_encodings.input_ids.size(1)
    test_labels = LabelProcess(test_labels, length, decoder)
    TestDataset = BertNerDataset(test_encodings, test_labels)

    torch.save(TrainDataset, 'data//liu_data//albert-TrainDataset.pt')
    torch.save(ValDataset, 'data//liu_data//albert-ValDataset.pt')
    torch.save(TestDataset, 'data//liu_data//albert-TestDataset.pt')


def GLUE_process(pt):
    """
    GLUE NER数据集的处理
    """
    df = pd.read_json(pt, lines=True)
    contents = df.text.to_list()
    df_label = df.label.to_list()
    labels = [['O'][::]*len(i) for i in contents]
    for i in range(len(labels)):
        for key in df_label[i].keys():
            for value in df_label[i][key].values():
                value = value[0]
                labels[i][value[0]] = 'B-'+key
                labels[i][value[0]+1:value[1]+1] = ['I-'+key][::]*(value[1]-value[0])

    tokenizer = BertTokenizer.from_pretrained("albert_chinese_base")
    encodings = tokenizer(contents[::], padding=True, truncation=True, max_length=256, return_tensors="pt")
    length = encodings.input_ids.size(1)
    decoder = DECODER().get('glue_e2t')
    labels = LabelProcess(labels, length, decoder)
    Dataset = BertNerDataset(encodings, labels)
    return Dataset


def data_augment(times=4):
    augment_dict = defaultdict(list)

    df = pd.read_json('cluener_public/train.json', lines=True)
    contents = df.text.to_list()
    df_label = df.label.to_list()
    labels = [['O'][::] * len(i) for i in contents]
    for i in range(len(labels)):
        for key in df_label[i].keys():
            for value in df_label[i][key].values():
                value = value[0]
                labels[i][value[0]] = 'B-' + key
                labels[i][value[0] + 1:value[1] + 1] = ['I-' + key][::] * (value[1] - value[0])
    # 上述循环生成了每段文本对应的BIO序列，每个字符一一对应

    contents_with_no_tags = []
    for i, text in enumerate(contents):
        start = 0
        end = 0
        for j, tag in enumerate(labels[i]):
            if j < len(labels[i])-1:
                if tag[0] == 'O':
                    end += 1
                else:
                    if end - start > 1:
                        contents_with_no_tags.append(text[start:end])
                    start, end = j+1, j+1
            else:
                if tag[0] == 'O':
                    contents_with_no_tags.append(text[start:])
    # 这段代码将每段文本，在实体处截断

    for str in contents_with_no_tags:
        augment_dict[len(str)].append(str)
        phrases = list(jieba.cut(str, cut_all=True))
        for p in phrases:
            augment_dict[len(p)].append(p)
    augment_dict = {key:list(set(augment_dict[key])) for key in augment_dict.keys()}
    # 通过结巴分词，利用原数据集文本，生成了非实体的扩充文本字典，
    # 后续可以拿来替换那些实体完成数据扩充

    augment_contents, augment_labels = [], []
    positions = []
    for i in range(len(df_label)):
        positions.append(
            reduce(lambda x,y:x+y, [pos for sub_dict in df_label[i].values() for pos in sub_dict.values()])
        )


    for i, pos in enumerate(positions):
        for _ in range(times):
            aug_con = contents[i][::]
            aug_lab = labels[i][::]
            for p in pos:
                if random.uniform(0,1) > 0.2:
                    aug_con = aug_con[:p[0]] + \
                              augment_dict[p[1]+1-p[0]][random.sample(range(len(augment_dict[p[1]+1-p[0]])),1)[0]] + \
                              aug_con[p[1]+1:]
                    aug_lab[p[0]:p[1]+1] = ['O'][::]*(p[1]+1-p[0])
            augment_contents.append(aug_con)
            augment_labels.append(aug_lab)

    contents += augment_contents
    labels += augment_labels

    tokenizer = BertTokenizer.from_pretrained("albert_chinese_base")
    encodings = tokenizer(contents[::], padding=True, truncation=True, max_length=256, return_tensors="pt")
    length = encodings.input_ids.size(1)
    decoder = DECODER().get('glue_e2t')
    labels = LabelProcess(labels, length, decoder)
    Dataset = BertNerDataset(encodings, labels)

    return contents, labels, Dataset


if __name__ == '__main__':
    # liu_process()
    # TrainDataset = GLUE_process(pt='cluener_public/train.json')
    # torch.save(TrainDataset, 'cluener_public/albert-train.pt')
    # ValDataset = GLUE_process(pt='cluener_public/dev.json')
    # torch.save(ValDataset, 'cluener_public/albert-dev.pt')

    contents, labels, AugTrainDataset = data_augment(times=4)
    torch.save(AugTrainDataset, 'cluener_public/albert-train-aug.pt')