import numpy as np
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config


def generate_line_dict_from_ner_data(ner_data_file):
    line_dict = {}
    sentence_count = 0
    line_number = 0
    for line in open(ner_data_file):
        inputs = line.strip().split()
        if len(inputs) >= 2:
            line_number += 1
            # word, ner_label, re_label, cve_id, tc, link = inputs
            # line_dict[line_number] = [word, ner_label, re_label, cve_id, tc, link]
            line_dict[line_number] = inputs
        else:
            if line_number in line_dict:
                line_dict[line_number].append('')

            sentence_count += 1
            line_number = config.ner_max_len * sentence_count
    return line_dict


def label_decode(label):
    if label == 'O':
        return 'O', 'O'
    return label[0], label[1]


def process(word):
    word = word.lower()
    word = "".join(c if not c.isdigit() else '0' for c in word)
    return word


def cnt_line(filename):
    line_cnt = 0
    cur_flag = False
    for line in open(filename):
        if line.strip() == '':
            if cur_flag:
                line_cnt += 1
            cur_flag = False
            continue
        cur_flag = True
    if cur_flag: line_cnt += 1
    return line_cnt


def create_word_index(filenames):
    word_index, word_cnt = {}, 1

    for sign, filename in enumerate(filenames):
        for line in open(filename):
            if line.strip() == '': continue
            word = line.strip().split()[0]
            #word = process(word)
            if word in word_index: continue
            word_index[word] = word_cnt
            word_cnt += 1
    return word_index, word_cnt


def create_char_index(filenames):
    char_index, char_cnt = {}, 3

    for filename in filenames:
        for line in open(filename):
            if line.strip() == '': continue
            word = line.strip().split()[0]
            for c in word:
                if c not in char_index:
                    char_index[c] = char_cnt
                    char_cnt += 1
    return char_index, char_cnt


def read_data(filename, word_index):
    line_cnt = cnt_line(filename)
    x, y = np.zeros((line_cnt, config.ner_max_len), dtype = np.int32), np.zeros((line_cnt, config.ner_max_len), dtype = np.int32)
    mask = np.zeros((line_cnt, config.ner_max_len), dtype = np.float32)
    i, j = 0, 0
    for line in open(filename):
        inputs = line.strip().split()
        if len(inputs) < 2:
            if j > 0:
                i, j = i + 1, 0
            continue
        if j == config.ner_max_len:
            continue
        word, label = inputs[0], inputs[1]
        # word = process(word)
        '''
        if word in word_index:
            word_ind = word_index[word]
        else:
            word_index[word] = word_cnt
            word_cnt += 1

            word_ind = word_index[word]
        label_ind = commons.LABEL_INDEX.index(label)
        x[i, j] = word_ind
        y[i, j] = label_ind
        mask[i, j] = 1.0
        j += 1
        '''
        if word not in word_index: continue
        word_ind = word_index[word]
        label_ind = config.labels.index(label)
        x[i, j] = word_ind
        y[i, j] = label_ind
        mask[i, j] = 1.0
        j += 1
    return x, y, mask


def read_char_data(filename, char_index):
    line_cnt = cnt_line(filename)
    x = np.zeros((line_cnt, config.ner_max_len, config.ner_max_len_char), dtype = np.int32)
    mask = np.zeros((line_cnt, config.ner_max_len, config.ner_max_len_char), dtype = np.float32)
    i, j = 0, 0
    for line in open(filename):
        inputs = line.strip().split()
        if len(inputs) < 2:
            if j > 0:
                i, j = i + 1, 0
            continue
        if j == config.ner_max_len:
            continue
        word, label = inputs[0], inputs[1]
        for k, c in enumerate(word):
            if k + 1 >= config.ner_max_len_char: break
            '''
            if c in char_index:
                x[i, j, k + 1] = char_index[c]
            else:
                char_index[c] = char_cnt - 2
                char_cnt += 1
                x[i, j, k + 1] = char_index[c]            
            '''
            if c not in char_index: continue
            x[i, j, k + 1] = char_index[c]
            mask[i, j, k + 1] = 1.0
        x[i, j, 0] = 1
        mask[i, j, 0] = 1.0
        if len(word) + 1 < config.ner_max_len_char:
            x[i, j, len(word) + 1] = 2
            mask[i, j, len(word) + 1] = 1.0
        j += 1
    return x, mask


def extract_ent(y, m):
    ret = []
    i = 0
    # pdb.set_trace()
    while i < y.shape[0]:
        # pdb.set_trace()
        if m[i] == 0:
            i += 1
            continue
        c1, c2 = label_decode(config.labels[y[i]])
        if c1 == 'O':
            i += 1
            continue
        if c1 == 'S':
            ret.append((i, i + 1, c2))
            i += 1
            continue
    return ret


def read_word2embedding():
    words = []
    for line in open(config.hash_file):
        words.append(line.strip())
    word2embedding = {}
    for i, line in enumerate(open(config.emb_file)):
        inputs = line.strip().split()
        word2embedding[words[i]] = np.array([float(e) for e in inputs], dtype = np.float32)
    return word2embedding


if __name__ == '__main__':
    generate_line_dict_from_ner_data('/Users/yingdong/Desktop/release/dataset/ner_data/sqli_test.txt')
    # generate_line_dict_from_ner_data('/Users/yingdong/Desktop/vulnerability_data/project_data/ner_re_dataset/ner_data_input/sqli_test.txt_cut_199')
