from lxml import html
import subprocess
import yaml

from data_collection.data_cleaning import write_report_for_ner
from data_collection.report_crawler import crawl_content_from_link


def get_unstructured_version_dict(dict_to_write):
    write_report_for_ner(dict_to_write)
    # todo: test this
    unstructured_version_dict = dict()
    for cve_id in dict_to_write:
        vul_cateory = get_vul_category_from_cvedetails(cve_id)
        # pdb.set_trace()
        merge_dict_cve_as_key(ner_re(vul_cateory), unstructured_version_dict)
    return unstructured_version_dict


def ner_re(vul_category):
    ner_model_name = ner_model_prefix + model_idx
    re_model_name = re_model_prefix + model_idx

    ner_model_path_and_name = ner_model_dir + ner_model_name
    re_model_path_and_name = re_model_dir + re_model_name

    # if ner_model_name not in os.listdir(ner_model_dir):
    #     print('ERROR! ner model does not exist!')
    # if re_model_name not in os.listdir(re_model_dir):
    #     print('ERROR! re model does not exist!')

    write_sh_str = get_sh_write_str(ner_model_path_and_name, re_model_path_and_name)
    with open(sh_name, 'w') as f_write:
        f_write.write(write_sh_str)
    subprocess.call('./end_to_end.sh')
    # pair_dict = get_version_data(tmp_data_dir, re_output_name)
    pair_dict = get_dict(tmp_data_dir, re_output_name)
    return pair_dict


def get_sh_write_str(ner_model_path_and_name, re_model_path_and_name):
    write_sh_str = '#!/bin/sh\n\n'
    write_sh_str += 'python ' + ner_test_script_path_and_name + \
                    ' --web_show True --ner_input ' + ner_input + \
                    ' --ner_output ' + ner_output + \
                    ' --ner_model ' + ner_model_path_and_name + \
                    ' --gazetteer False' + \
                    ' --gru ' + gru + '\n\n'
    write_sh_str += 'source activate tensorflow\n\n'
    write_sh_str += 'python ' + re_test_script_path_and_name + \
                    ' --web_show True --ner_output ' + ner_output + \
                    ' --re_input ' + re_input + \
                    ' --re_output ' + re_output + \
                    ' --re_output_dict ' + re_output_dict + \
                    ' --re_model ' + re_model_path_and_name + \
                    ' --gru ' + gru + '\n\n'
    write_sh_str += 'source deactivate tensorflow\n\n'
    return write_sh_str


def get_dict(tmp_data_dir, re_output_name):
    pair_dict = {}
    with open(tmp_data_dir + re_output_name) as f_read:
        content = f_read.read()
        if '{' in content:
            pair_dict = yaml.load(content[content.find('=')+2:])
    return pair_dict


def merge_dict_cve_as_key(dict1, dict2):
    for key1 in dict1:
        dict2[key1] = dict1[key1]


def get_vul_category_from_cvedetails(cve_id):
    # https://www.cvedetails.com/vulnerability-search.php?f=1&vendor=&product=&cveid=CVE-2014-0413&msid=&bidno=&cweid=&cvssscoremin=&cvssscoremax=&psy=&psm=&pey=&pem=&usy=&usm=&uey=&uem=
    prefix = 'https://www.cvedetails.com/vulnerability-search.php?f=1&vendor=&product=&cveid='
    suffix = '&msid=&bidno=&cweid=&cvssscoremin=&cvssscoremax=&psy=&psm=&pey=&pem=&usy=&usm=&uey=&uem='
    searh_page_link = prefix + cve_id + suffix
    matched_vul_category = extract_vul_category_from_return_page(searh_page_link)
    return matched_vul_category


def extract_vul_category_from_return_page(link):
    crawl_result = crawl_content_from_link(link)
    if crawl_result is None:
        return 1
    raw_content, clean_content = crawl_result
    tree = html.fromstring(raw_content)
    vul_type = tree.xpath('//*[@id="vulnslisttable"]/tr[2]/td[5]/text()')
    if len(vul_type) > 0:
        print(len(vul_type), vul_type)
        vul_type = vul_type[0].lower()
        vul_type = vul_type.replace('\n', '').replace('\t', '').strip()

    type_dict = {'mem. corr.': 0,
                 'bypass': 1,
                 'dir. trav.': 2,
                 'file inclusion': 3,
                 'dos': 4,
                 'exec code': 5,
                 '+priv': 6,
                 'http r.spl.': 7,
                 '+info': 8,
                 'csrf': 9,
                 'overflow': 10,
                 'sql': 11,
                 'xss': 12,
                 '': 1
                 }

    for type_ in type_dict:
        if type_ in vul_type:
            return type_dict[type_]
    return 0
