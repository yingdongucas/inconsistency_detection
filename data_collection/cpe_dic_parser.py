import os, sys, inspect

from xml.dom import minidom

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import utils

windows_version_list = []


def parse_cpe_xml(cpe_dic_name):
    software_version_dict = dict()
    xmldoc = minidom.parse(cpe_dic_name)
    itemlist = xmldoc.getElementsByTagName('cpe-23:cpe23-item')
    for s in itemlist:
        cpe = s.attributes['name'].value
        software, version = get_software_name_and_version_from_cpe(cpe)
        software = clean_software_name(software)
        if software not in software_version_dict:
            software_version_dict[software] = []
        if software and version:
            software_version_dict[software].append(version)
    print('len(software_version_dict): ', len(software_version_dict))
    write_software_name_version_dict(software_version_dict)


def get_software_name_and_version_from_cpe(cpe):
    # print(cpe)
    parts = cpe.split(':')[2:]
    # print(parts)
    software, version = '', ''

    num_found = False
    idx = 0

    for part in parts:
        if not utils.contain_letter(part) and not utils.contain_number(part):
            break
        part = part.replace('_', ' ').replace('~', ' ')

        if part[0].isdigit():
            version += part + ' '
            num_found = True
        elif num_found:
            version += part + ' '
        else:
            software += part + ' '

        idx += 1

    software = software.strip()
    version = version.strip()
    if version == '':
        software, version = extract_windows_version(software)
    if software == '':
        software, version = corner_case(version)

    return software, version


def extract_windows_version(software):
    keyword = 'microsoft windows'
    if software.startswith(keyword):
        return keyword, software.replace(keyword, '').strip()
    return software, ''


def corner_case(version):
    # '1024cms 1024 cms 1.4.2 beta'
    version_split = version.split()
    version_split.reverse()

    idx = 0
    for word in version_split:
        if utils.contain_number(word):
            version_split.reverse()
            return ' '.join(version_split[:-idx-1]), ' '.join(version_split[-idx-1:])
        idx += 1
    return '', version


def write_software_name_version_dict(software_version_dict):
    for software in software_version_dict:
        software_version_dict[software] = sorted(software_version_dict[software])
    with open('cpe_name_dic.py', 'w') as f_write:
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
    cpe_dic = 'PATH_TO_CPE/official-cpe-dictionary_v2.3.xml'       # modify the path
    parse_cpe_xml(cpe_dic)
