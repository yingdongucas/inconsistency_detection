import re
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils
from data_collection.utils_DATA import encode_content, get_software_from_cpe_dic


def extract_pair_from_structured_reports(cve_id, report_category, target_content, version_dict):
    pair_dict = dict()
    if report_category == 2:
        pair_dict = extract_pair_from_securityfocus_official(target_content['content'])
    elif report_category == 3:
        pair_dict = extract_pair_from_securitytracker(cve_id, target_content, version_dict)
    else:
        print('ERROR in extract_pair!')
    return pair_dict


def extract_pair_from_securityfocus_official(report_content):
    report_content_list = report_content.split('\n')
    pair_dict = dict()
    for line in report_content_list:
        if line.strip() == '':
            continue
        content_line_version, content_line_software = get_pair_from_content_line_focus_official(line)
        if content_line_software != '' and content_line_version != '':
            if content_line_software not in pair_dict:
                pair_dict[content_line_software] = []
            if content_line_version != '0' and content_line_version not in pair_dict[content_line_software]:
                pair_dict[content_line_software].append(content_line_version)
    return pair_dict


def extract_pair_from_securitytracker(cve_id, report_content, version_dict):
    pair_dict = dict()
    cve_id_list, title, content = report_content['cve_id'], report_content['title'], report_content['content']
    print(cve_id, 111, cve_id_list)
    cve_id_list = list(set([cve_id] + cve_id_list))
    title = title.lower()
    content = content.lower()
    if content != '':
        software_set = get_title_software_from_other_db(cve_id_list, title, version_dict)
        for software in software_set:
            pair_dict[software] = [content]
    return pair_dict


def extract_pair_from_edb_title(raw_title):
    title_dict = {}
    if raw_title.find(' - ') != -1:
        # software_in_title might contain version
        software_in_title = raw_title.split(' - ')[0]
        contains_number = utils.contain_number(software_in_title)
        if contains_number:
            content_line_version, content_line_software = get_pair_from_content_line_focus_official(
                raw_title)
            content_line_version = content_line_version[:content_line_version.find(' - ')]
            content_line_version, content_line_software = move_range_from_software_to_version(content_line_version, content_line_software)
            title_dict = {encode_content(content_line_software): encode_content(content_line_version)}
    return title_dict


def get_title_software_from_other_db(cveid_list, title_this_db, version_dict):
    # todo: get title software from gazetteer; test with http://www.securitytracker.com/id/1041303
    title_this_db = title_this_db.lower()
    candidate_software_in_title_set = set()
    for cveid in cveid_list:
        if cveid not in version_dict:
            continue
        for db_type in config.report_list:
            if db_type == 'securitytracker' or db_type not in version_dict[cveid]:
                continue
            for link_other_db in version_dict[cveid][db_type]:
                for tc in version_dict[cveid][db_type][link_other_db]:
                    for software in version_dict[cveid][db_type][link_other_db][tc]:
                        if software in title_this_db:
                            candidate_software_in_title_set.add(software)
    candidate_software_in_title_set = remove_substr_in_set(candidate_software_in_title_set)
    if candidate_software_in_title_set == set():
        candidate_software_in_title_set = get_software_from_cpe_dic(title_this_db)
    return candidate_software_in_title_set


def remove_substr_in_set(str_set):
    new_set = set()
    for strr in str_set:
        if set_strs_not_contain(strr, str_set):
            strr = strr.strip(' c')
            new_set.add(strr)
    return new_set


def set_strs_not_contain(substr, str_set):
    for strr in str_set:
        if strr == substr:
            continue
        if substr in strr:
            return False
    return True


def get_pair_from_content_line_focus_official(line):
    line = line.lower()
    con_word_list = line.split()
    content_line_version, content_line_software = '', ''
    keyword_software_loc = 0
    # contains keyword software
    word_idx = 0
    mat1 = False
    version_loc = -1
    for word in con_word_list:
        if word == '':
            continue

        if word in ['windows', 'office']:
            version_loc = word_idx + 1
            mat1 = True
            break
        word_idx += 1
    if mat1:
        content_line_version = get_right_part(con_word_list, version_loc)
        content_line_software = get_left_part(con_word_list, version_loc)
    else:
        mat1 = False
        # the n-th word is version
        version_loc = 0
        word_idx = 0
        for word in con_word_list:
            if word == '':
                continue
            mat1 = re.match(r'(v)?[\d]{1,2}((\.[\d]{1,2}){1,2}(\.x)?|\.x)', word)
            if mat1:
                version_loc = word_idx
                break
            word_idx += 1

        # contains 1.1.x format number
        if version_loc != 0:
            content_line_version = get_right_part(con_word_list, version_loc)
            content_line_software = get_left_part(con_word_list, version_loc)
            # content_line_software = con[:con.find(content_line_version)].strip()
            # print(content_line_software, ' ||| ', content_line_version)
            # print(con)
            # print()

        # find word that is a number
        else:
            version_loc = 0
            word_idx = 0
            for word in con_word_list:
                if word == '':
                    continue
                contains_number = utils.contain_number(word)
                if contains_number and word.lower not in ['x64', 'x86', 'x86_64']:
                    version_loc = word_idx
                    break
                word_idx += 1
            content_line_version = get_right_part(con_word_list, version_loc)
            content_line_software = get_left_part(con_word_list, version_loc)
            # print(content_line_software, ' ||| ', content_line_version)
            # print(con)
            # print()
            if content_line_version in content_line_software:
                print('ERROR if content_line_version in content_line_software:')
    return content_line_version.lower(), remove_duplicate_word_from_software(content_line_software).lower()


def get_right_part(word_list, loc):
    idx = 0
    right_str = ''
    for word in word_list:
        if idx >= loc:
            right_str += word + ' '
        idx += 1
    return right_str.strip()


def get_left_part(word_list, loc):
    idx = 0
    right_str = ''
    for word in word_list:
        if idx < loc:
            right_str += word + ' '
        idx += 1
    return right_str.strip()


def remove_duplicate_word_from_software(content_line_software):
    word_list = content_line_software.split()
    new_word_list = []
    idx = 0
    for word in word_list:
        if idx - 1 >= 0:
            if word_list[idx - 1] != word:
                new_word_list.append(word)
        else:
            new_word_list.append(word)
        idx += 1
    new_software = utils.format_str(new_word_list, sep=' ')
    # if content_line_software != new_software:
    #     print(new_software, '|', content_line_software)
    return new_software


def move_range_from_software_to_version(content_line_version, content_line_software):
    # {'smartermail <': '7.2.3925'}
    new_software = ''
    range_start_flg = False
    content_line_version = ' ' + content_line_version
    for ch in content_line_software:
        if ch in ['<', '>', '=']:
            range_start_flg = True
        if range_start_flg:
            content_line_version = ch + content_line_version
        else:
            new_software += ch

    return content_line_version.strip(), new_software.strip()