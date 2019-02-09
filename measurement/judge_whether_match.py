from match_software_names import get_matched_software_names
from version_data_normalization import format_version
from version_data_clearning import remove_letters_from_version_list
from version_mapping import map_range_to_points
from loose_match import loose_compare_version_point


def compare_standard_and_db_version_dict(cve_id, db, standard_content_dict, version_dict_db, loose, debug_mode=False):
    matched_software_dict = get_matched_software_names(list(standard_content_dict.keys()),
                                                       list(version_dict_db.keys()))
    software_match_dict = dict()
    software_direction_list = []
    for cve_software in matched_software_dict:
        matched_db_software_list = matched_software_dict[cve_software]
        software_match_dict[cve_software] = False
        for db_software in matched_db_software_list:
            cve_version = standard_content_dict[cve_software]
            db_version = version_dict_db[db_software]
            compare_version_result = compare_version_list(cve_id, db,
                                           cve_software, db_software,
                                           cve_version, db_version)

            version_same, software_direction = None, None
            if type(compare_version_result) == bool:
                version_same = compare_version_result
            else:
                version_same, software_direction = compare_version_result

            if version_same:
                software_match_dict[cve_software] = True
                software_direction_list.append(software_direction)
                continue

    for cve_software in software_match_dict:
        if not software_match_dict[cve_software]:
            if loose:
                return False, ''
            return False

    if not loose:
        return True

    report_direction = compute_report_direction_from_software_direction_list(software_direction_list)
    # important!
    if debug_mode:
        if report_direction in ['under', 'over', 'both']:
            if len(version_dict_db) == 1:
                report_direction = 'under'
            # print('direction: ' + report_direction)
            print(cve_id, db, report_direction)
            print(standard_content_dict)
            print(version_dict_db)
            print()
    if report_direction is None:
        if loose:
            return True, 'exact'
        return True
    return True, report_direction


def compute_report_direction_from_software_direction_list(software_direction_list):
    # if None in software_direction_list:
    #     print(software_direction_list)
    if 'under' in software_direction_list and 'over' in software_direction_list:
        return 'both'
    if 'under' in software_direction_list:
        return 'under'
    if 'over' in software_direction_list:
        return 'over'
    return None


def compare_version_list(cve_id, db, software_1, software_2, version_list_1, version_list_2, loose):

    if version_list_1 == version_list_2:
        return True

    converted_range_version_list_1 = format_version(cve_id, db, software_1, software_2, version_list_1)
    converted_range_version_list_2 = format_version(cve_id, db, software_1, software_2, version_list_2)

    if converted_range_version_list_1 == converted_range_version_list_2:
        return True
    # print(cveid, db)

    only_digits_version_list_1 = remove_letters_from_version_list(converted_range_version_list_1)
    only_digits_version_list_2 = remove_letters_from_version_list(converted_range_version_list_2)

    version_point_set_1 = map_range_to_points(only_digits_version_list_1, software_1)
    version_point_set_2 = map_range_to_points(only_digits_version_list_2, software_2)

    strict_match_result = False
    if version_point_set_1 == version_point_set_2:
        strict_match_result = True

    loose_match_result = loose_compare_version_point(cve_id, db, version_point_set_1, version_point_set_2)

    if type(loose_match_result) == bool:
        strict_match_result = loose_match_result

    if loose:
        return loose_match_result
    else:
        return strict_match_result


