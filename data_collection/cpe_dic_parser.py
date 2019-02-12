import os, sys, inspect

from xml.dom import minidom

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils


def parse_cpe_xml():
    software_version_dict = dict()
    cpe_dic_name = config.cpe_dic_path + 'official-cpe-dictionary_v2.3.xml'
    xmldoc = minidom.parse(cpe_dic_name)
    itemlist = xmldoc.getElementsByTagName('cpe-23:cpe23-item')
    # print(len(itemlist))
    # print(itemlist[0].attributes['name'].value)
    # print(commons.excel_data_path.replace('_a', ''))
    with utils.add_path('/Users/yingdong/Desktop/vulnerability/measurement'):
        module = __import__('nvd_parser')
        for s in itemlist:
            cpe = s.attributes['name'].value
            software, version = module.get_software_name_and_version_from_cpe(cpe)
            software = clean_software_name(software)
            if software.startswith('a '):
                print(software)
            if software not in software_version_dict:
                software_version_dict[software] = []
            if version != '':
                software_version_dict[software].append(version)
    print('len(software_version_dict): ', len(software_version_dict))
    write_software_name_version_dict(software_version_dict)


def write_software_name_version_dict(software_version_dict):
    for software in software_version_dict:
        software_version_dict[software] = sorted(software_version_dict[software])
    with open(config.cpe_dic_path + 'cpe_name_dic.py', 'w') as f_write:
        f_write.write('cpe_software_version_dict = ' + str(software_version_dict))


def clean_software_name(software):
    software_word_list = software.split()
    if software_word_list[0] in ['a', 'h', 'o']:
        software_word_list = software_word_list[1:]
        software = ' '.join(software_word_list)
    if len(software_word_list) > 1 and software_word_list[0] == software_word_list[1]:
        software_word_list = software_word_list[1:]
        software = ' '.join(software_word_list)
    return software


if __name__ == '__main__':
    parse_cpe_xml()
