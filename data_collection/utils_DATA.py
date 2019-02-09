import re
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils


def encode_content(content):
    return str(content.encode('utf-8')).strip("b' ").strip("'").replace('\t', ' ').replace('\\t', ' ').replace('\\n', '\n').strip()


def regex_cve(str_):
    p = re.compile(r'CVE-\d+-\d+')
    target = p.findall(str_)
    return target


def sorted_by_value(dict_1):
    return sorted(dict_1.items(), key=lambda kv: kv[1])


def get_software_from_cpe_dic(title_this_db):
    with utils.add_path(config.cpe_dic_path):
        category_module = __import__(config.cpe_dic_file)
        cpe_software_version_dict = category_module.cpe_software_version_dict
        candidate_software_in_title_set = set()
        title_word_list = title_this_db.split()
        for cpe_software in cpe_software_version_dict.keys():
            cpe_software_word_list = cpe_software.split()
            if all(i in title_word_list for i in cpe_software_word_list):
                candidate_software_in_title_set.add(cpe_software)
        return candidate_software_in_title_set


def get_cve_id_list(title, clean_content):
    cve_id_list = regex_cve(title)
    if cve_id_list == []:
        cve_id_list = regex_cve(clean_content)
    return cve_id_list