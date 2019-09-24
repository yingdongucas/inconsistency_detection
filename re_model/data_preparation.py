import numpy as np
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils


def separate_label_same_entity_into_single_entity(original_list):
    current_idx = original_list[0]
    new_list = [[current_idx]]
    for entity_idx in original_list[1:]:
        if entity_idx == current_idx + 1:
            new_list[-1].append(entity_idx)
            current_idx = entity_idx
        else:
            new_list.append([entity_idx])
            current_idx = entity_idx
    return new_list


def generate_re_data_for_ner_output(ner_data_path_and_name, re_data_path_and_name, fixlen):
    f_read = open(ner_data_path_and_name, 'r')
    f_write = open(re_data_path_and_name, 'w')

    split_sents = f_read.read().split('\n\n')

    software_label = config.software_label
    version_label = config.version_label

    for sent in split_sents:
        if sent.find(software_label) != -1 and sent.find(version_label) != -1:
            split_sent = sent.split('\n')
            software_loc_dic = dict()
            version_loc_dic = dict()
            software_loc_list = []
            version_loc_list = []
            word_list = []
            word_idx = 0
            cve_db = None
            for word_line in split_sent:
                if word_idx == 0:
                    cve_db = word_line[word_line.find(' CVE') + 1:]
                word_line_split = word_line.split()
                word = word_line_split[0]
                ner_label = word_line_split[1]
                word_list.append(word)
                if ner_label == software_label:
                    software_loc_list.append(word_idx)
                    software_loc_dic[word_idx] = word
                elif ner_label == version_label:
                    version_loc_list.append(word_idx)
                    version_loc_dic[word_idx] = word
                word_idx += 1

            entity_sent_list = get_entity_idx_and_new_sent(software_loc_list, version_loc_list, word_list)
            for entity_sent in entity_sent_list:
                entity_start_1, entity_start_2, new_sent = entity_sent
                write_re_data_line(entity_start_1, entity_start_2, fixlen, 'n', new_sent, cve_db, f_write)
    f_read.close()
    f_write.close()


def get_entity_idx_and_new_sent(software_loc_list, version_loc_list, word_list, sep='_'):
    all_software_phrase_loc_list = get_entity_phrase_idx_list(software_loc_list)
    all_version_phrase_loc_list = get_entity_phrase_idx_list(version_loc_list)

    entity_sent_list = []
    # format: entity_1_idx, entity_2_idx, new_sentence

    for one_software_phrase_loc_list in all_software_phrase_loc_list:
        for one_version_phrase_loc_list in all_version_phrase_loc_list:

            new_sentence = ''
            software_appear = False
            version_appear = False
            software_idx = 0
            version_idx = 0
            word_idx = 0
            original_word_idx = 0
            for word in word_list:
                if original_word_idx in one_software_phrase_loc_list:

                    if not software_appear:
                        software_idx = word_idx
                        software_appear = True
                        new_sentence += ' ' + word
                        word_idx += 1
                    else:
                        new_sentence += sep + word

                elif original_word_idx in one_version_phrase_loc_list:

                    if not version_appear:
                        version_idx = word_idx
                        version_appear = word_idx
                        new_sentence += ' ' + word
                        word_idx += 1
                    else:
                        new_sentence += sep + word

                else:

                    new_sentence += ' ' + word
                    word_idx += 1
                original_word_idx += 1

            entity_sent_list.append([software_idx, version_idx, new_sentence.strip()])
    return entity_sent_list


def get_entity_phrase_idx_list(loc_list):
    all_phrase_loc_list = []
    one_phrase_loc_list = []
    ll = list(range(loc_list[-1] + 1))
    for idx in ll:
        if idx in loc_list:
            one_phrase_loc_list.append(idx)
            if idx + 1 not in loc_list:
                all_phrase_loc_list.append(one_phrase_loc_list)
                one_phrase_loc_list = []
    return all_phrase_loc_list


def write_re_data_line(entity_start_1, entity_start_2, fixlen, relation, new_sent, cve_db, f_write):
    judge_result = judge_str_satisy_fixlen(entity_start_1, entity_start_2, fixlen, relation, new_sent.strip())

    if judge_result is not False:
        [entity_start_1, entity_start_2, new_sent] = judge_result
        if cve_db.find('_') == -1:
            cve_db = cve_db.replace(' ', '_').strip('_')
            write_str = str(entity_start_1) + ' ' + str(entity_start_2) + ' ' + relation + ' ' + str(
                new_sent).strip() + ' ' + cve_db.strip() + '\n'
            f_write.write(write_str)
        else:
            print('ERROR in cve_db')


def judge_str_satisy_fixlen(en1pos, en2pos, fixlen, relation, sentence):

    relation = config.relation2id[relation]
    sentence = sentence.split()
    # if not satisfy, return False, else return the new [en1pos, en2pos, sentence]
    new_sentence = []
    if en1pos >= fixlen or en2pos >= fixlen:
        if relation != 1:
            return False
        center_pos = int((en1pos + en2pos) / 2)
        new_sentence = sentence[int(center_pos - fixlen / 2): int(center_pos + fixlen / 2)]
        en1pos -= int(center_pos - fixlen / 2)
        en2pos -= int(center_pos - fixlen / 2)

        en1pos = int(en1pos)
        en2pos = int(en2pos)

        if en1pos < 0 or en2pos < 0 or en2pos >= fixlen or en1pos >= fixlen:
            return False

    final_sentence = []
    if new_sentence != []:
        final_sentence = new_sentence
    else:
        final_sentence = sentence

    en1_en2_appear_1 = 0
    for i in range(fixlen):
        if i in [en1pos, en2pos]:
            en1_en2_appear_1 += 1
            continue
    if en1_en2_appear_1 != 2:
        print('error append pos  ', en1_en2_appear_1, relation, en1pos, en2pos)
        return False

    en1_en2_appear_2 = 0
    for i in range(min(fixlen, len(final_sentence))):
        if i in [en1pos, en2pos]:
            en1_en2_appear_2 += 1
            continue
    if en1_en2_appear_2 != 2:
        print('error append word  ', en1_en2_appear_2, relation, en1pos, en2pos)
        return False

    final_sentence = utils.transform_list_to_str(final_sentence)

    return [en1pos, en2pos, final_sentence]


def generate_entity_str_using_entity_idx(list_of_entity_list, original_sent_loc, remove_duplicate_entities = True):
    separate_entities_str_list = []
    for entity_list in list_of_entity_list:
        entity_str = ''
        for entity_idx in entity_list:
            entity_str += original_sent_loc[entity_idx] + ' '
        if remove_duplicate_entities:
            if entity_str.strip() in separate_entities_str_list:
                pass
            else:
                separate_entities_str_list.append(entity_str.strip())
        else:
            separate_entities_str_list.append(entity_str.strip())
    return separate_entities_str_list


def generate_prediction_and_gt_table(source_data_txt_file, relations=None, dup=False):
    f_source_data_txt_file = open(source_data_txt_file, 'r')
    content_lines = f_source_data_txt_file.readlines()
    name_version_dict = {}
    if relations is not None:
        if type(relations) == np.ndarray:
            relations=relations.tolist()
    idx = -1
    for content in content_lines:
        idx += 1
        content = content.strip().split()
        relation = content[2]

        if relations is not None:
            relation = config.id2relation[relations[idx]]
        if relation == 'n':
            continue

        en1pos = int(content[0])
        en2pos = int(content[1])
        
        sentence = content[3:]
        if dup:
            sentence = content[3:-1]
        cve_db = content[-1].replace('_', ' ')

        if cve_db not in name_version_dict:
            name_version_dict[cve_db] = {}

        en1 = ''
        en2 = ''

        for i in range(len(sentence)):
            if i == en1pos:
                en1 = sentence[i]
            if i == en2pos:
                en2 = sentence[i]

        if en1 in name_version_dict[cve_db]:
            if en2 not in name_version_dict[cve_db][en1]:
                name_version_dict[cve_db][en1].append(en2)
        else:
            name_version_dict[cve_db][en1] = [en2]
    f_source_data_txt_file.close()
    if dup:
        new_name_version_dict = get_report_level_name_version_dict(name_version_dict)
        return new_name_version_dict


def get_report_level_name_version_dict(name_version_dict):
    tc_dict = {'t': 'title', 'c': 'content'}
    report_name_version_dict = {}
    for cve_tc_link in name_version_dict:
        cve_id, tc, db_link = cve_tc_link.split()
        db, link = db_link.split('|')
        if db == '' or link == '':
            print('ERROR 1')
            return
        if cve_id not in report_name_version_dict:
            report_name_version_dict[cve_id] = {}

        if db not in report_name_version_dict[cve_id]:
            report_name_version_dict[cve_id][db] = {}

        if link not in report_name_version_dict[cve_id][db]:
            report_name_version_dict[cve_id][db][link] = {'title': {}, 'content': {}}
        elif report_name_version_dict[cve_id][db][link]['title'] != {}:
            if db == 'title':
                print('ERROR 2')
                return
        elif report_name_version_dict[cve_id][db][link]['content'] != {}:
            if db == 'content':
                print('ERROR 3')
                return

        if report_name_version_dict[cve_id][db][link][tc_dict[tc]] == {}:
            report_name_version_dict[cve_id][db][link][tc_dict[tc]] = name_version_dict[cve_tc_link]
        else:
            print('ERROR 4')
            return
    return report_name_version_dict


# if __name__ == '__main__':
#     dir1 = '/Users/yingdong/Desktop/vulnerability_data/project_data/ner_re_dataset/ner_data_input/memc_full_duplicate'
#     dir2 = dir1 + '.txt_cut_199'
#     generate_re_data_for_ner_output(dir2, dir1, config.re_max_len)
