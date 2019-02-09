
import nltk
import re
import string
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils


def write_report_for_ner(dict_to_write):
    # report content --> ner data format
    transfer_data_file = open(config.tmp_data, 'w')
    target_dataset = dict_to_write
    for cve_id in target_dataset:
        db_sent = target_dataset[cve_id]
        for db in db_sent:
            # if db in ['securitytracker', 'securityfocus_official']:
            #     continue
            for link in db_sent[db]:
                for title_or_content in db_sent[db][link]:

                    # ******************* consistent with a_get_200_cveid_and_generate_report_data.py ****************
                    clean_sents = db_sent[db][link][title_or_content].replace('\n', ' ')
                    clean_sents = re.sub('(https?://[^\s]+)', ' ', clean_sents)

                    # deal with avast!
                    if db == 'cve' and clean_sents.find('!') != -1:
                        clean_sents = clean_sents.replace('! ', ' ')

                    clean_sents = clean_sents.replace('SiT!', 'SiT').replace('Ver. 1.20', 'version 1.20')

                    for replace_str in ['\\n', '\\r', "b'", '\\t', '\\', '\xa0']:
                        clean_sents = clean_sents.replace(replace_str, ' ')
                    #
                    # # wrong
                    for remove_str in ['%', '=', '-', '#', '<', '_', '>', ' ']:
                        clean_sents = re.sub(remove_str + '{2,}', ' ', clean_sents)
                    clean_sents = re.sub('(> )+', ' ', clean_sents)
                    clean_sents = re.sub(' +', ' ', clean_sents)

                    dict_cve_db_link_sents = nltk.sent_tokenize(clean_sents)
                    for start_str in [' Confirmed', ' Summary:', 'Reported by', 'Exploit Title', 'Underlying OS',
                                      ' We ',
                                      ' On ', ' Tested', ' Vendor', 'Comment', ' Hi ', ' Hi,', ' Subject:', ' Fixed',
                                      ' Affect', ' Description:', ' Vulnerable', ' Not vulnerable', ' Non-vulnerable',
                                      ' Versions fixed: ', 'Versions affected: ',
                                      ' vulnerable versions: ', ' vulnerable version: ',
                                      ]:
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
                    sents = set()
                    for sent in dict_cve_db_link_sents:
                        sent = sent.strip()

                        # mat1 = False
                        # for s in sent.split():
                        # mat1 = re.match(r'(v)?[\d]{1,2}((\.[\d]{1,2}){1,2}(\.x)?|\.x)', s)
                        # if mat1:
                        #     break
                        contains_letter = utils.contain_letter(sent)
                        find_space = (sent.find(' ') != -1)
                        if find_space and contains_letter and (len(sent) > 3):
                            sents.add((sent + title_or_content[0]).strip('> '))
                    # print('load data success')
                    # ******************* consistent with a_get_200_cveid_and_generate_report_data.py ****************

                    # splits = nltk.sent_tokenize(db_sent[db][link][title_or_content])
                    # for s in dict_cve_db_link_sents:
                    #     sents.add(s)

                    # sents = set(db_sent[db])
                    write_sents(sents, transfer_data_file, cve_id, db, link)
    transfer_data_file.close()
    # transfer_data_file = open(tmp_data, 'r')
    # lines = transfer_data_file.read()
    # if lines.find('\n\n\n') != -1:
    #     f_data = open(ner_input + '_new', 'w')
    #     f_data.write(lines.replace('\n\n\n', '\n\n'))
    #     f_data.close()
    transfer_data_file.close()

    transfer_data_file = open(config.tmp_data, 'r')
    lines = transfer_data_file.read()
    lines_split = lines.split('\n\n')

    f_data = open(config.ner_input, 'w')
    write_str = ''
    for sent in lines_split:
        sent = sent.strip('\n')
        sent_split = sent.split('\n')
        sent_len = 0
        for word_line in sent_split:
            write_str += word_line + '\n'
            sent_len += 1
            if sent_len == 198:
                break
        write_str += '\n'
    f_data.write(write_str)
    transfer_data_file.close()
    remove_tmp_data()
    print('ner data generation done...')
    # return sentence_list, temp_list2
    # pass


def remove_tmp_data():
    os.remove(config.tmp_data)


def clean_sentence(sentence):
    for ss_remove in ['"', '*', '#', '[0]', '[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]', '?', '//',
                      '/ ', '> >', "'s", "'"]:
        sentence = sentence.replace(ss_remove, ' ')

    sentence = sentence.replace("/'", "'")

    # sentence = re.sub('> [a-zA-Z]', ' ', sentence)
    for alphabet in list(string.ascii_uppercase + string.ascii_lowercase):
        sentence = sentence.replace('> ' + alphabet, alphabet)

    sentence = re.sub('([^a-zA-Z0-9]{7,})', ' ', sentence)

    # always last
    sentence = re.sub(' +', ' ', sentence).strip()
    return sentence


def write_sents(sents, transfer_data_file, cve_id, db, link):
    for sent in sents:
        sent = sent.strip("'")
        sent, t_or_c = sent[:-1], sent[-1]
        sent = sent.strip()

        if not sent.startswith('Vendor URL'):
            sent = clean_sentence(sent)
            write_transfer_data_file_simple_text(transfer_data_file, sent, '',
                                                 with_cve_db=True, cve=cve_id, title_or_content=t_or_c,
                                                 db=db + '|' + link)
            transfer_data_file.write('\n')


def write_transfer_data_file_simple_text(transfer_data_file, transfer_content, type_value,  with_cve_db=False, cve='', title_or_content='c', db=''):
    append_str = ''
    if with_cve_db:
        append_str = ' ' + cve + ' ' + title_or_content + ' ' + db

    words = nltk.word_tokenize(transfer_content)
    idx = 0
    for ww in words:
        if utils.is_ascii(ww) and (not ww.startswith('www.')):
            if type_value in ['vulnerable_']:
                if ww == 'Access':
                    transfer_data_file.write(ww + ' ' + 'S-' + type_value + 'software' + ' O' + append_str + '\n')
                else:
                    transfer_data_file.write(ww + ' ' + 'O' + ' O' + append_str + '\n')
            else:
                transfer_data_file.write(ww + ' ' + 'O' + ' O' + append_str + '\n')
        idx += 1
