import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
import numpy as np

from copy import deepcopy
import pdb
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils

from data_preparation import generate_re_data_for_ner_output


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


def reverse_dict(word2id):
    id2word = dict()
    for word in word2id:
        id_ = word2id[word]
        id2word[id_] = word
    return id2word


def enrich_char2id(char2id, word):
    for char in word:
        if char in char2id or ord(char) >= 128:
            continue
        char2id[char] = len(char2id)


def read_word_embedding():
    logging.info('reading word embedding data...')
    vec = []
    word2id = {}
    char2id = {}
    f = open(config.word_emb_path_and_name, encoding='utf-8')
    dim = config.word_emb_dim_re
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        if len(content) != dim + 1:
            break
        word = content[0]
        word2id[word] = len(word2id)
        enrich_char2id(char2id, word)
        content = content[1:]
        content = [float(i) for i in content]
        vec.append(content)
    f.close()

    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    char2id['UNK'] = len(char2id)
    char2id['BLANK'] = len(char2id)

    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))

    vec = np.array(vec, dtype=np.float32)
    np.save(config.vec_npy_path_and_name, vec)

    id2word = reverse_dict(word2id)
    with open(config.word2id_file_path_and_name, 'w') as f_write:
        f_write.write('word2id = ' + str(word2id))
    with open(config.id2word_file_path_and_name, 'w') as f_write:
        f_write.write('id2word = ' + str(id2word))
    with open(config.char2id_file_path_and_name, 'w') as f_write:
        f_write.write('char2id = ' + str(char2id))


# reading data
def init_test_data(re_origin_test_data_name_read, generate_from_ner_output_path=None, category_separate_idx=None):
    word2id = utils.get_word2id()
    logging.info('reading test data ...')

    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
    gt_y_list = []
    origin_test_data_path_and_name = config.labeled_re_data_input_path + re_origin_test_data_name_read
    if generate_from_ner_output_path is not None:
        # if category_separate_idx is None:
            origin_test_data_path_and_name = generate_from_ner_output_path + re_origin_test_data_name_read
            # origin_test_data_path_and_name = origin_test_data_path_and_name.replace('_gaze', '')
            generate_re_data_for_ner_output(origin_test_data_path_and_name + config.data_suffix, origin_test_data_path_and_name, config.re_max_len)
        # else:
        #     origin_test_data_path_and_name = generate_from_ner_output_path + re_origin_test_data_name_read + '_' + str(category_separate_idx)
        #     ner_data_path_and_name = generate_from_ner_output_path + re_origin_test_data_name_read + commons.ner_data_suffix + '_' + str(category_separate_idx)
        #     generate_re_data_for_ner_output(ner_data_path_and_name, origin_test_data_path_and_name, commons.fixlen)
    print('test data:', origin_test_data_path_and_name)
    f = open(origin_test_data_path_and_name, 'r', encoding='utf-8')
    count = 0
    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split()

        en1pos = int(content[0])
        en2pos = int(content[1])

        relation = config.relation2id[content[2]]

        sentence = content[3:-1]
        en1 = ''
        en2 = ''

        for i in range(len(sentence)):
            if i == en1pos:
                en1 = sentence[i]
            if i == en2pos:
                en2 = sentence[i]

        tup = (en1, en2, count)
        count += 1

        if tup not in test_sen:
            test_sen[tup] = []
            y_id = relation
            label = [0 for i in range(len(config.relation2id))]
            label[y_id] = 1
            test_ans[tup] = label
        else:
            y_id = relation
            test_ans[tup][y_id] = 1

        output = []
        
        en1_en2_appear = 0
        for i in range(config.re_max_len):
            # logging.info(i)
            if i in [en1pos, en2pos]:
                en1_en2_appear += 1
                continue
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])
        if en1_en2_appear != 2:
            logging.info('error append pos test ' + str(en1_en2_appear) + ' ' + str(en1pos) + ' ' + str(en2pos) + ' ' + str(relation))

        en1_en2_appear = 0
        for i in range(min(config.re_max_len, len(sentence))):
            if i in [en1pos, en2pos]:
                en1_en2_appear += 1
                continue
            word = 0
            if sentence[i] not in word2id:
                ps = None
                if '**' in sentence[i]:
                    ps = sentence[i].split('**')
                else:
                    ps = sentence[i].split('_')
                # avg_vec = np.zeros(commons.word_emb_dim)
                c = 0
                for p in ps:
                    if p in word2id:
                        c += 1
                        # avg_vec += vec[word2id[p]]
                if c > 0:
                    # avg_vec = avg_vec / c
                    word2id[sentence[i]] = len(word2id)
                    # vec.append(avg_vec)
                else:
                    word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i - en1_en2_appear][0] = word
        if en1_en2_appear != 2:
            logging.info('error append word test ' + str(en1_en2_appear) + ' ' + str(en1pos) + ' ' + str(en2pos) + ' ' + str(relation))

        gt_y_list.append(relation)
        test_sen[tup].append(output)

    test_x = []
    test_y = []

    for i in test_sen:
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])

    test_x_npy = np.array(test_x)
    test_y_npy = np.array(test_y)

    return test_x_npy, test_y_npy, origin_test_data_path_and_name


def init_train_data(re_origin_train_data_name_read):
    word2id = utils.get_word2id()
    train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector

    logging.info('reading train data...')
    origin_train_data_path_and_name = config.labeled_re_data_input_path + re_origin_train_data_name_read
    f = open(origin_train_data_path_and_name, 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split()
        # get entity name
        en1pos = int(content[0])
        en2pos = int(content[1])
        relation = config.relation2id[content[2]]

        sentence = content[3:-1]

        en1 = ''
        en2 = ''

        for i in range(len(sentence)):
            if i == en1pos:
                en1 = sentence[i]
            if i == en2pos:
                en2 = sentence[i]

        tup = (en1, en2)

        if tup not in train_sen:
            train_sen[tup] = []
            train_sen[tup].append([])
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(config.relation2id))]
            label[y_id] = 1
            train_ans[tup] = []
            train_ans[tup].append(label)
        else:
            y_id = relation
            label = [0 for i in range(len(config.relation2id))]
            label[y_id] = 1

            temp = find_index(label, train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                label_tag = len(train_ans[tup]) - 1
                train_sen[tup].append([])
            else:
                label_tag = temp

        output = []
        en1_en2_appear_1 = 0
        for i in range(config.re_max_len):
            if i in [en1pos, en2pos]:
                en1_en2_appear_1 += 1
                continue
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])
        if en1_en2_appear_1 != 2:
            logging.info('error append pos ' + str(en1_en2_appear_1) + ' ' + str(en1pos) + ' ' + str(en2pos) + ' ' + str(relation))

        en1_en2_appear_2 = 0
        for i in range(min(config.re_max_len, len(sentence))):
            if i in [en1pos, en2pos]:
                en1_en2_appear_2 += 1
                continue
            word = 0
            if sentence[i] not in word2id:
                ps = sentence[i].split('_')
                # avg_vec = np.zeros(commons.word_emb_dim)
                c = 0
                for p in ps:
                    if p in word2id:
                        c += 1
                        # avg_vec += vec[word2id[p]]
                if c > 0:
                    # avg_vec = avg_vec / c
                    word2id[sentence[i]] = len(word2id)
                    #vec.append(avg_vec)
                else:
                    word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            output[i-en1_en2_appear_2][0] = word

        if en1_en2_appear_2 != 2:
            logging.info('error append word ' + str(en1_en2_appear_2) + ' ' + str(en1pos) + ' ' + str(en2pos) + ' ' + str(relation))

        train_sen[tup][label_tag].append(output)

    train_x = []
    train_y = []

    for i in train_sen:
        if len(train_ans[i]) != len(train_sen[i]):
            logging.info('ERROR')
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])

    train_x_npy = np.array(train_x)
    train_y_npy = np.array(train_y)

    return train_x_npy, train_y_npy


def separate_new(train_test_x_y_npy_name):

    logging.info('reading training data')
    train_test_x_y_npy_path_and_name = config.labeled_re_data_write_path + train_test_x_y_npy_name
    x_train = np.load(train_test_x_y_npy_path_and_name + '_x.npy')

    # x_train = np.load('./new_category_data/train_x.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []

    logging.info('seprating train data')
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_train[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    word_npy = np.array(train_word)
    pos1_npy = np.array(train_pos1)
    pos2_npy = np.array(train_pos2)
    return word_npy, pos1_npy, pos2_npy


def separate(x_npy):
    train_test_word = []
    train_test_pos1 = []
    train_test_pos2 = []

    logging.info('seprating data')
    for i in range(len(x_npy)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_npy[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_test_word.append(word)
        train_test_pos1.append(pos1)
        train_test_pos2.append(pos2)

    train_test_char = get_char_data_from_word_data(train_test_word)

    char_npy = np.array(train_test_char)
    word_npy = np.array(train_test_word)
    pos1_npy = np.array(train_test_pos1)
    pos2_npy = np.array(train_test_pos2)
    return char_npy, word_npy, pos1_npy, pos2_npy


def read_char2id():
    with utils.add_path(config.labeled_re_data_write_path):
        category_module = __import__('char2id_file')
        char2id = category_module.char2id
        return char2id


def read_id2word():
    with utils.add_path(config.labeled_re_data_write_path):
        category_module = __import__('id2word_file')
        id2word = category_module.id2word
        return id2word


def get_char_data_from_word_data(word_data):
    char_data = deepcopy(word_data)
    char2id = read_char2id()
    id2word = read_id2word()
    entity_idx = 0
    for entity_sentence in char_data:
        sentence_idx = 0
        for sentence in entity_sentence:
            char_data[entity_idx][sentence_idx] = get_char_list_of_word_list(sentence, char2id, id2word)
            sentence_idx += 1
        entity_idx += 1

    return char_data


def get_char_list_of_word_list(word_list, char2id, id2word):
    char_list_of_sentence = []
    for word in word_list:
        char_list_of_word = get_char_of_word(word, char2id, id2word)
        char_list_of_sentence.append(char_list_of_word)
    return char_list_of_sentence


def get_word_from_id2word(word_idx, id2word):
    if word_idx in id2word:
        return id2word[word_idx]
    return 'UNK'


def get_char_of_word(word, char2id, id2word):
    char_list = []
    word = get_word_from_id2word(word, id2word)
    for char in word:
        if len(char_list) == config.re_max_len_char:
            break
        if char not in char2id:
            char_list.append(char2id['UNK'])
        else:
            char_list.append(char2id[char])
    char_list = pad_char_list(char_list, char2id)
    return char_list


def pad_char_list(char_list, char2id):
    while len(char_list) < config.re_max_len_char:
        char_list.append(char2id['BLANK'])
    return char_list


def judge_wordchar2id_vec_npy_exists():
    if not os.path.exists(config.word2id_file_path_and_name) \
            or not os.path.exists(config.char2id_file_path_and_name) \
            or not os.path.exists(config.id2word_file_path_and_name) \
            or not os.path.exists(config.vec_npy_path_and_name):
        read_word_embedding()


def generate_train_npy_data(category):
    judge_wordchar2id_vec_npy_exists()
    train_data = category + '_train' + config.data_suffix

    train_x_npy, train_y_npy = init_train_data(train_data)
    train_char_npy, train_word_npy, train_pos1_npy, train_pos2_npy = separate(train_x_npy)

    return train_y_npy, train_char_npy, train_word_npy, train_pos1_npy, train_pos2_npy


def generate_test_npy_data(category, duplicate=False, generate_from_ner_output_path=None, category_separate_idx=None, gaze=False):
    judge_wordchar2id_vec_npy_exists()

    test_flg = None
    # if generate_from_ner_output_path == commons.unlabeled_ner_data_output_path:
    #     test_flg = '_full'
    # else:
    #     test_flg = '_test'

    test_flg = '_test'

    test_data_name = category + test_flg
    # if duplicate:
    #     test_data_name = category + test_flg + '_duplicate'
    if gaze:
        test_data_name += '_gaze'

    test_data_name += config.data_suffix

    test_x_npy, test_y_npy, test_txt_file_path_and_name = init_test_data(test_data_name, generate_from_ner_output_path, category_separate_idx=category_separate_idx)
    test_char_npy, test_word_npy, test_pos1_npy, test_pos2_npy = separate(test_x_npy)

    return test_y_npy, test_char_npy, test_word_npy, test_pos1_npy, test_pos2_npy, test_txt_file_path_and_name, test_data_name


if __name__ == '__main__':
    # generate_train_npy_data(transfer=False)
    for cc in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        cat = config.num_cat_dict[cc]
        generate_test_npy_data(cat, duplicate=False)
