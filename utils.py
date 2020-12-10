import re
import pickle
import pandas as pd
import torch


def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()


def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent


def labeling(text, entity):
    """
    :param text: list(str) 例如['旺旺贷这个应用很火。','恒生公司是主题。']
    :param entity: list(list(str)) 例如[['旺贷','旺旺贷'],['恒生','恒生公司']]
    :return: list(list(str)) 例如[['B','I','I','O',...'O'],['B','I','I','I','O',...'O']]
    """
    label = [['O'][::] * len(str(t)) for t in text]
    for i, l in enumerate(label):
        t = text[i]
        et = entity[i]
        et.sort(key=lambda x: len(x))
        for e in et:
            if e:
                for j in range(len(t) - len(e) + 1):
                    if t[j:j + len(e)] == e:
                        l[j] = 'B'
                        l[j + 1:j + len(e)] = ['I'][::] * (len(e) - 1)
    return label


def LabelProcess(labels, length, decoder):
    # decoder = {'O': 0, 'B': 1, 'I': 2}
    # decoder = DECODER().get('liu_e2t')
    for l in labels:
        l[:] = [0] + [decoder[c] for c in l]
        if len(l) < length:
            l.extend([0][::] * (length - len(l)))
        else:
            l[:] = l[:length]
    return labels


def read_xlsx(xlsx_dir, col, tp='str'):
    """
    :param xlsx_dir: str
    :param col: list
    :return: list
    读取excel文档将多条新闻转化为列表，匹配bert tokenizer格式 list(str)
    """
    df = pd.read_excel(io=xlsx_dir, usecols=col)
    data = df.values.tolist()
    if tp == 'str':
        content = [str(d[0]) for d in data]
    elif tp == 'int':
        content = [int(d[0]) for d in data]
    else:
        content = [d[0] for d in data]
    return content


def compute_precision(tags, labels):
    """
    :param tags: torch.tensor (batch_size, seq_len, num_labels)
    :param labels: torch.tensor (batch_size, seq_len, num_labels)
    :return: torch.tensor (1)
    """
    try:
        return int(torch.sum((tags != 0) * (tags == labels))) / int(torch.sum(tags != 0))
    except:
        return 0


def compute_recall(tags, labels):
    """
    :param tags: torch.tensor (batch_size, seq_len, num_labels)
    :param labels: torch.tensor (batch_size, seq_len, num_labels)
    :return: torch.tensor (1)
    """
    try:
        return int(torch.sum((labels != 0) * (tags == labels))) / int(torch.sum(labels != 0))
    except:
        return 0


def raw2sentence(pt):
    df = pd.read_table(pt, header=None, names=['contents', 'labels'], sep='\t', skip_blank_lines=False)
    contents = df['contents'].tolist()
    labels = df['labels'].tolist()

    contents = [c if type(c) != float else ' ' for c in contents]
    labels = [c+' ' if type(c) != float else '  ' for c in labels]

    contents = (''.join(contents)).split(' ')
    labels = (''.join(labels)).split('   ')
    labels = [i.split() for i in labels]

    return contents, labels


def tensor2entity(T,decoder):
    # decoder = {0: 'O',
    #            1: 'B-PER',
    #            2: 'I-PER',
    #            3: 'B-ORG',
    #            4: 'I-ORG',
    #            5: 'B-LOC',
    #            6: 'I-LOC',
    #            }
    entity = []
    for t in T:
        entity.append([decoder[int(i)] for i in t])
    return entity


def split_entity(label_sequence):
    entity_mark = dict()
    entity_pointer = None
    for index, label in enumerate(label_sequence):
        if label.startswith('B'):
            category = label.split('-')[1]
            entity_pointer = (index, category)
            entity_mark.setdefault(entity_pointer, [label])
        elif label.startswith('I'):
            if entity_pointer is None: continue
            if entity_pointer[1] != label.split('-')[1]: continue
            entity_mark[entity_pointer].append(label)
        else:
            entity_pointer = None
    return entity_mark


def evaluate(LABELS, TAGS):
    """
    LABELS和TAGS的格式范例：
    [
    ['O','B-PER','I-PER','O','O'],
    ['O','B-PER','I-PER','O','O','B-LOC','I-LOC','O'],
    ]
    """
    real_entity_num = 0
    predict_entity_num = 0
    true_entity_num = 0
    for i in range(len(TAGS)):
        real_label = LABELS[i]
        predict_label = TAGS[i]
        real_entity_mark = split_entity(real_label)
        predict_entity_mark = split_entity(predict_label)

        true_entity_mark = dict()
        key_set = real_entity_mark.keys() & predict_entity_mark.keys()
        for key in key_set:
            real_entity = real_entity_mark.get(key)
            predict_entity = predict_entity_mark.get(key)
            if tuple(real_entity) == tuple(predict_entity):
                true_entity_mark.setdefault(key, real_entity)

        real_entity_num += len(real_entity_mark)
        predict_entity_num += len(predict_entity_mark)
        true_entity_num += len(true_entity_mark)

    precision = true_entity_num / predict_entity_num if predict_entity_num != 0 else 0
    recall = true_entity_num / real_entity_num if real_entity_num != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, f1


class DECODER():
    def __init__(self,):
        self.all_decoder = dict()
        self.all_decoder['liu_e2t'] = {
            'O': 0,
            'B-PER': 1,
            'I-PER': 2,
            'B-ORG': 3,
            'I-ORG': 4,
            'B-LOC': 5,
            'I-LOC': 6,
        }

        self.all_decoder['liu_t2e'] = {self.all_decoder['liu_e2t'][key]:key for key in self.all_decoder['liu_e2t'].keys()}

        self.all_decoder['glue_e2t'] = {
            'O': 0,
            'B-address': 1,
            'I-address': 2,
            'B-book': 3,
            'I-book': 4,
            'B-company': 5,
            'I-company': 6,
            'B-game': 7,
            'I-game': 8,
            'B-government': 9,
            'I-government': 10,
            'B-movie': 11,
            'I-movie': 12,
            'B-name': 13,
            'I-name': 14,
            'B-organization': 15,
            'I-organization': 16,
            'B-position': 17,
            'I-position': 18,
            'B-scene': 19,
            'I-scene': 20,
        }

        self.all_decoder['glue_t2e'] = {self.all_decoder['glue_e2t'][key]:key for key in self.all_decoder['glue_e2t'].keys()}

    def get(self, type):
        return self.all_decoder[type]