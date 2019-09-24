import requests
import nltk
from bs4 import BeautifulSoup
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils


def get_software_name_links():
    software_start_char_list = get_software_start_chars()
    print(software_start_char_list)

    software_start_char_link_dict = dict()
    # format: {char: {'sha': XXX, 'page_num': 32}, ...}

    for start_char in software_start_char_list:
        software_start_char_link_dict[start_char] = {'sha': '', 'page_num': 0}

    sha_trc_dic = assign_sha_to_star_char_link()
    return sha_trc_dic


def crawl_software_list(raw_cvedetails_software_list_file_name):
    software_company_list = []
    # format: [[software 1, company 1], [software 2, company 2], ...]

    sha_trc_dic = get_software_name_links()
    page_num_dic = dict()

    for start_char in sha_trc_dic:
        page_num_dic[start_char] = 1
        page_idx = 1
        while True:
            if page_idx > page_num_dic[start_char]:
                break
            link = 'https://www.cvedetails.com/product-list/product_type-/vendor_id-0/firstchar-' + start_char + '/page-' + str(page_idx) + '/products.html?' + sha_trc_dic[start_char]
            print('char ' + start_char + ', page ' + str(page_idx) + ' ' + link)

            page = requests.get(link, timeout=60, headers={'User-Agent': "Magic Browser"})
            content = BeautifulSoup(page.content).get_text()
            keyword_section = content.split('\n')
            keyword_section = [x.strip('\t').strip() for x in keyword_section if x != ''][132:]

            if 'function copypaging(){' in keyword_section:
                max_page = keyword_section[keyword_section.index('function copypaging(){') - 1]
                if max_page == '(This Page)':
                    page_num_dic[start_char] = 1
                elif max_page.startswith('(This Page)') and max_page[-1] != ')':
                    page_num_dic[start_char] = int(max_page[-1])
                else:
                    page_num_dic[start_char] = int(max_page)

            line_idx = 0
            while True:
                line_list = keyword_section[line_idx: line_idx+7]
                software_company = line_list[:2]
                print(software_company)
                software_company_list.append(software_company)
                line_idx += 8

                if page_idx == 1 and line_idx >= len(keyword_section):
                    break
                elif line_idx >= len(keyword_section) or keyword_section[line_idx].startswith('Total number of products'):
                    break
            page_idx += 1
    print(page_num_dic)
    print(len(software_company_list))

    with open(raw_cvedetails_software_list_file_name + '.py', 'w') as f_write:
        f_write.write('software_company_list = ' + str(software_company_list))
    return software_company_list


def contain_illegal_chars(software_company):
    software_tokenized = nltk.sent_tokenize(software_company[0])
    company_tokenized = nltk.sent_tokenize(software_company[1])
    if len(software_tokenized) > 1 or len(company_tokenized) > 1:
        return True
    return False


def with_both_dots_and_numbers(software_company):
    contain_a_word_with_both_digits_and_numbers = False
    software, company = software_company
    software_contain_a_word_with_both_digits_and_numbers = judge_word_in_sentence_with_both_dots_and_numbers(software)
    company_contain_a_word_with_both_digits_and_numbers = judge_word_in_sentence_with_both_dots_and_numbers(company)
    # if software_contain_a_word_with_both_digits_and_numbers:
    #     print(software)
    # if company_contain_a_word_with_both_digits_and_numbers:
    #     print(company)
    if software_contain_a_word_with_both_digits_and_numbers or company_contain_a_word_with_both_digits_and_numbers:
        return True
    return False


def judge_word_in_sentence_with_both_dots_and_numbers(sent):
    words = nltk.word_tokenize(sent)
    for word in words:
        contains_number = utils.contain_number(word)
        contains_dot = '.' in word
        if contains_number and contains_dot:
            return True
    return False


def clean_software_company(software_company):
    software, company = software_company
    software = remove_last_dot_word(software)
    company = remove_last_dot_word(company)
    return [software, company]


def remove_last_dot_word(company):
    company = company.lower()
    company_split = company.split()
    if company_split[-1].endswith('.'):
        company = utils.format_str(company_split[:-1], sep=' ')
    return company


def clean_software_company_list(clean_cvedetails_software_file_name, add_version_dict_labeled_software=True):

    # if raw_cvedetails_software_list file is not in current dir, add the dir to sys path and use __import__ as follows:
    # module = __import__(raw_cvedetails_software_list_file_name.replace('.py', ''))
    # software_company_list = module.software_company_list

    from raw_cvedetails_software_list import software_company_list

    software_name_list = []
    software_clean_cnt = 0
    company_clean_cnt = 0
    for software_company in software_company_list:

        # contain ! or other symbols that start a new sentence
        contain_illegal = contain_illegal_chars(software_company)

        # remove software names where a word both contain dots and numbers
        contain_a_word_with_both_dots_and_numbers = with_both_dots_and_numbers(software_company)

        if not contain_illegal and not contain_a_word_with_both_dots_and_numbers:
            software, company = clean_software_company(software_company)

            # with or without company name
            if software != '':
                software_name_list.append(software)
                software_clean_cnt += 1
            if software != company:
                software_with_company = (company + ' ' + software).strip()
                if software_with_company != '':
                    software_name_list.append(software_with_company)
                    company_clean_cnt += 1

    software_name_list = list(set(software_name_list))
    print('raw software name cnt: ', len(software_company_list))
    print('clean software name cnt: ', software_clean_cnt)
    print('clean company name cnt: ', company_clean_cnt)
    print('final clean software name cnt: ', len(software_name_list))

    # if add_version_dict_labeled_software:
    #     from measurement.clean_version_and_measure import get_all_cat_version_dict, get_software_freq_from_version_dict
    #     all_cat_dict = get_all_cat_version_dict(config.version_dict_file_path, commons.labeled_version_dict_file_name)
    #     software_name_with_freq_dic = get_software_freq_from_version_dict(get_from_version_dict=True,
    #                                                                       get_from_report_content_freq=False,
    #                                                                       all_cat_version_dict=all_cat_dict)
    #     labeled_software_name_list = list(software_name_with_freq_dic.keys())
    #     for i in labeled_software_name_list:
    #         if i not in software_name_list:
    #             software_name_list.append(i)
    #             print(i)
    #     print('software name cnt with labeled software added: ', len(software_name_list))

    with open(clean_cvedetails_software_file_name + '.py', 'w') as f_write:
        f_write.write('clean_software_name_list = ' + str(software_name_list))
    return software_name_list


def get_software_start_chars():
    start_char_str = '. ( @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'
    software_start_char_list = start_char_str.split()
    software_start_char_list = [i.lower() for i in software_start_char_list]
    return software_start_char_list


def assign_sha_to_star_char_link():
    # sha_dic = dict()
    # for c in software_start_char_list:
    #     sha_dic[c] = ''
    sha_trc_dic = {'.': 'sha=cbf15ace5a04b5bb7231641bef0c8748e010b454&trc=10&order=1',
                   '(': 'sha=f97ceaf7c1422961d8ca25ebf6de67a6a2aa85ef&trc=1&order=1',
                   '@': 'sha=46d1857ac75b18d69489962330398cf3dbd7b9dc&trc=8&order=1',
                   '0': 'sha=a8ad92bfaa0f9e6d7cf0e21b9ca2c8ed910e261b&trc=11&order=1',
                   '1': 'sha=5ea19524fc8d447e81b4611de9281cf6c7d8c725&trc=213&order=1',
                   '2': 'sha=0794cb225083924b8922b87987547edb334eacd2&trc=94&order=1',
                   '3': 'sha=32e44df2809b52479aa72c9d7bd85a2655e7ab17&trc=162&order=1',
                   '4': 'sha=dabdd7c49448e9815c01b643bad5002b9cf00874&trc=105&order=1',
                   '5': 'sha=6c19d8d0d8f29d0b95049026578e0c4518627ddc&trc=116&order=1',
                   '6': 'sha=30e0c6c3575bc5c80f5e789c5221c3d645cfceda&trc=57&order=1',
                   '7': 'sha=4ab0ac9df9acd78013f2d5c076ff1c443b75314f&trc=108&order=1',
                   '8': 'sha=19aa91eb208261005928ec2d47d0b012c0ced7a7&trc=100&order=1',
                   '9': 'sha=3ff12eaffeac96b9fef7be66c04affe6786893e7&trc=25&order=1',
                   'a': 'sha=167446a54c46896db65c7fcae78596325eef719f&trc=3240&order=1',
                   'b': 'sha=f59a3ae552c9c076beb97f2f6a9acb4543d94c6a&trc=1490&order=1',
                   'c': 'sha=5778c5af94fc66f7737e3f3841ca94a2b92a53c4&trc=3804&order=1',
                   'd': 'sha=2ed9c4ac439c8d07b431029129caf941a7d5d371&trc=2195&order=1',
                   'e': 'sha=74f52c8152a0ae12e938664e06e4347010efcff5&trc=2107&order=1',
                   'f': 'sha=97ebb23b48d07fc53510628aa1b2537f22e6b798&trc=1786&order=1',
                   'g': 'sha=6dd5dc74ad79442e5d96a4ff9e3fd6d771a8cdfa&trc=1121&order=1',
                   'h': 'sha=b34d6c09424ced0b88309ceff6ca81f689d9ec3e&trc=1241&order=1',
                   'i': 'sha=f3950f83f43a7d227014d0f60674e305c1e05099&trc=2034&order=1',
                   'j': 'sha=77c45c37644b3d6f9a0902851146a979bd882500&trc=864&order=1',
                   'k': 'sha=081c433333c44b2cf21380263b29e7006f7f6c56&trc=526&order=1',
                   'l': 'sha=5ace52c0e88558a707726f1072a77363182a8055&trc=1512&order=1',
                   'm': 'sha=35b7afdbec9bb7da0f18e1f0ed7e6ff6f1c07de0&trc=2676&order=1',
                   'n': 'sha=56f242c37788988c8a13058d101ed9d88c4e1a4c&trc=1661&order=1',
                   'o': 'sha=14950a95fb301185c6fde0bdca8a941012524288&trc=1168&order=1',
                   'p': 'sha=90d7b31c349de2e72fc4096e086ec600791918a2&trc=3247&order=1',
                   'q': 'sha=37bdb376b489c9a2cf70e56368bbaad01aff465f&trc=337&order=1',
                   'r': 'sha=6ab0566b62b1c47ef90e6a77f89869a32d1f96b0&trc=1676&order=1',
                   's': 'sha=03046d1ccfe7201381dccc460c28e2683ed3ad40&trc=5055&order=1',
                   't': 'sha=ba8cef89df3178498a0a301f29387d375c6f01c9&trc=2372&order=1',
                   'u': 'sha=e6f2a10eaa509d570a6aebe55c34ac97ae21c28b&trc=725&order=1',
                   'v': 'sha=bb16df8d7df97896dcaaa6c26daea369c41022e5&trc=1015&order=1',
                   'w': 'sha=cdf6f99b7642c79677bce76d806911cd2bd9c4d0&trc=2136&order=1',
                   'x': 'sha=c37b4605b4d45106112c76ab5d7ed83a0d948a21&trc=885&order=1',
                   'y': 'sha=394e706f5e6ca0e97567d8c27c17dd215f146240&trc=219&order=1',
                   'z': 'sha=52191bf003282577cb340f95d986b09e129a7b96&trc=332&order=1'}
    return sha_trc_dic


if __name__ == '__main__':
    raw_cvedetails_software_list_file_name = ''           # specify the path where the generated
                                                          # raw software names crawled from cvedetails
                                                          # are located.

    clean_cvedetails_software_file_name = ''              # software name lists whose company names are removed.

    raw_cvedetails_software_list = crawl_software_list(raw_cvedetails_software_list_file_name)
    clean_cvedetails_software_list = clean_software_company_list(clean_cvedetails_software_file_name,
                                                                 add_version_dict_labeled_software=False)