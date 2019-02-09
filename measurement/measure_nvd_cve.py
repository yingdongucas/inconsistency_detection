

from judge_whether_match import compare_standard_and_db_version_dict
from version_data_clearning import clean_software_name_in_version_dict, clean_version_list


standard = 'nvd' # 'cve'
loose_ = True
debug_mode_ = False

modify_set = set()


def measure_version_dict(cve_id, version_dict_for_a_cve_id, loose, debug_mode=False, ignore_both_nvd_cve=False):
    ignore_list = ['vul_type', standard]
    if ignore_both_nvd_cve:
        ignore_list = ['vul_type', 'nvd', 'cve']
    standard_content_dict = {}
    # version_dict_for_a_cve_id = clean_software_name_in_version_dict(version_dict_for_a_cve_id)

    if standard in version_dict_for_a_cve_id:
        standard_link = list(version_dict_for_a_cve_id[standard].keys())[0]
        if 'content' in version_dict_for_a_cve_id[standard][standard_link]:
            standard_content_dict = version_dict_for_a_cve_id[standard][standard_link]['content']

    if standard_content_dict == {}:
        return standard_content_dict

    for db in version_dict_for_a_cve_id:
        if db in ignore_list:
            continue
        # if db != 'securitytracker':
        #     continue
        for link in version_dict_for_a_cve_id[db]:
            title_dict, content_dict = {}, {}
            if 'title' in version_dict_for_a_cve_id[db][link]:
                title_dict = version_dict_for_a_cve_id[db][link]['title']
                version_dict_for_a_cve_id[db][link].pop('title', None)
            if 'content' in version_dict_for_a_cve_id[db][link]:
                content_dict = version_dict_for_a_cve_id[db][link]['content']
                version_dict_for_a_cve_id[db][link].pop('content', None)

            tc = 't_c'
            version_dict_for_a_cve_id[db][link][tc] = merge_dict_software_as_key(title_dict, content_dict)
            for software in version_dict_for_a_cve_id[db][link][tc]:
                version_dict_for_a_cve_id[db][link][tc][software] = \
                    clean_version_list(db, version_dict_for_a_cve_id[db][link][tc][software])
            match_result = compare_standard_and_db_version_dict(cve_id, db, standard_content_dict, version_dict_for_a_cve_id[db][link][tc],
                                                                loose, debug_mode=debug_mode)
            match, report_direction = None, None
            if type(match_result) == bool:
                match = match_result
                version_dict_for_a_cve_id[db][link]['strict_match'] = match
            else:
                match, report_direction = match_result
                version_dict_for_a_cve_id[db][link]['loose_match'] = [match, report_direction]
    if standard == 'nvd':
        if 'cve' in version_dict_for_a_cve_id:
            del version_dict_for_a_cve_id['cve']
    else:

        if 'nvd' in version_dict_for_a_cve_id:
            del version_dict_for_a_cve_id['nvd']
    return version_dict_for_a_cve_id


def merge_dict_software_as_key(dict_1, dict_2):
    merged_dict = {}

    # cover dict1 contains and dict2 dict1 both contains
    for i in dict_1:
        if i in dict_2:
            merged_set = set(dict_1[i] + dict_2[i])
            merged_dict[i] = list(merged_set)
        else:
            merged_dict[i] = dict_1[i]

    # cover dict2 contains
    for i in dict_2:
        if i not in dict_1:
            merged_dict[i] = dict_2[i]
    return merged_dict


def normalize_software_name(software):
    # todo: enrich company name list
    # some software names are also company names, e.g., huawei. hehehhe
    property_name_list = ['module', 'system', 'software', 'component', 'function', 'plugin']
    clean_software = ''
    software_word_list = software.replace('_', ' ').split()
    if len(software_word_list) == 1:
        return software

    idx = 0
    for word in software_word_list:
        # if word in company_name_list:
        #     continue
        if idx == 0 or software_word_list[idx-1] != word:
            clean_software += word + ' '
        idx += 1

    return clean_software.strip()
