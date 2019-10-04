import os
import csv
import nltk
import sys
import re
import string

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


# data cleaning, convert reports into corpus to train word embeddings

def build_corpus(data_dir, cps_dir):
    num_cat_dict = {0: 'memc', 1: 'bypass', 2:'dirtra', 3:'fileinc', 4:'dos', 5:'execution', 6:'gainpre', 7:'httprs', 8:'infor', 9:'csrf', 10:'overflow', 11:'sqli', 12:'xss'}

    print(sys.path)
    sys.path.append(data_dir)
    sys.path = list(set(sys.path))
    print(sys.path)

    str_all_set = set()
    for category_num in list(range(13)):

        category = num_cat_dict[category_num]
        print(category)
        category_module = __import__(category)
        dict_to_write = category_module.dict_to_write

        for cve_id in dict_to_write:
            db_sent = dict_to_write[cve_id]
            for db in db_sent:
                if db in ['securitytracker', 'securityfocus_forum']:
                    continue
                for link in db_sent[db]:
                    for title_or_content in db_sent[db][link]:
                        sents = db_sent[db][link][title_or_content]
                        clean_sents = clean_str(sents)
                        for s in clean_sents:
                            str_all_set.add(s)
    str_all_set -= {''}

    clean_string = ''
    for s in str_all_set:
        clean_string += s.strip() + ' '
    with open(cps_dir + 'corpus.txt', 'w') as f_corpus:
        f_corpus.write(clean_string)


def clean_str(old_sents):
    clean_sents = old_sents.replace('\n', ' ')
    clean_sents = re.sub('(https?://[^\s]+)', ' ', clean_sents)
    # clean_sents = re.sub('(www\.[^\s]+)', ' ', clean_sents)

    # deal with avast!
    # if db == 'cve' and clean_sents.find('!') != -1:
    clean_sents = clean_sents.replace('! ', ' ')

    clean_sents = clean_sents.replace('SiT!', 'SiT').replace('Ver. 1.20', 'version 1.20')

    for replace_str in ['\\n', '\\r', "b'", '\\t', '\\', '\xa0']:
        clean_sents = clean_sents.replace(replace_str, ' ')

    for remove_str in ['%', '=', '-', '#', '<', '_', '>', ' ']:
        clean_sents = re.sub(remove_str + '{2,}', ' ', clean_sents)
    clean_sents = re.sub('(> )+', ' ', clean_sents)
    clean_sents = re.sub(' +', ' ', clean_sents)

    dict_cve_db_link_sents = nltk.sent_tokenize(clean_sents)
    for start_str in [' Confirmed', ' Summary:', 'Reported by', 'Exploit Title', 'Underlying OS', ' We ',
                      ' On     ', ' Tested', ' Vendor', 'Comment', ' Hi ', ' Hi,', ' Subject:', ' Fixed',
                      ' Affect', ' Description:', ' Vulnerable', ' Not vulnerable', ' Non-vulnerable']:
        idx = 0
        for sent in dict_cve_db_link_sents:
            start_loc = sent.find(start_str)
            if start_loc > 0:
                marked_sentence = sent.replace(start_str, '~!@#$%' + start_str)
                split_marked_sentence = marked_sentence.split('~!@#$%')
                indices = [i for i, x in enumerate(dict_cve_db_link_sents) if x == sent]
                for ii in indices:
                    dict_cve_db_link_sents[ii] = ''
                for ss in split_marked_sentence:
                    dict_cve_db_link_sents.append(ss)
            idx += 1
    sents = set()
    for sent in dict_cve_db_link_sents:

        for ss_remove in ['"', '*', '#', '[', ']', '?', '//', '/ ', '> >', "'s", "'"]:
            sentence = sent.replace(ss_remove, ' ')

        sentence = sentence.replace("/'", "'")

        for alphabet in list(string.ascii_uppercase + string.ascii_lowercase):
            sentence = sentence.replace('> ' + alphabet, alphabet)

        sentence = re.sub('([^a-zA-Z0-9]{7,})', ' ', sentence)

        # always last
        sentence = re.sub(' +', ' ', sentence).strip()

        contains_letter = any(i.isalpha() for i in sentence)
        find_space = (sentence.find(' ') != -1)
        if find_space and contains_letter and (len(sentence) > 3):
            sents.add(sent.strip('> '))
    return sents


if __name__ == '__main__':
    dataset_dir = 'DATASET_DIR'     # specify the path where the collected
                                    # reports are located -- should be consistent
                                    # with cvedetails_crawler.py

    corpus_dir = 'CORPUS_DIR'       # specify the path where the generated corpus.txt should be located
    build_corpus(dataset_dir, corpus_dir)


