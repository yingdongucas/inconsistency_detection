

def get_matched_software_names(cve_software_list, other_db_software_list):
    matched_software_dict = {}
    for cve_software in cve_software_list:
        matched_software_dict[cve_software] = {}
        exactly_same_software = ''
        partly_same_list = []
        for db_software in other_db_software_list:
            exactly_same_flg, db_software_or_false = software_name_is_the_same(cve_software, db_software)
            if exactly_same_flg:
                exactly_same_software = db_software
                break
            elif db_software_or_false is not False:
                partly_same_list.append(db_software)
            else:
                # todo: check not matched software
                pass
        if exactly_same_software != '':
            matched_software_dict[cve_software] = [exactly_same_software]
            continue
        elif partly_same_list != []:
            matched_software_dict[cve_software] = sorted(partly_same_list)
        else:
            del matched_software_dict[cve_software]
    return matched_software_dict


def software_name_is_the_same(software_1, software_2):
    # if any word in software_1 exists in software_2 (or vice versa), we consider it is the same
    # todo: double check this rule

    if software_1 == software_2:
        # exactly the same
        return True, software_2
    software_1_split = software_1.split()
    software_2_split = software_2.split()

    for w1 in software_1_split:
        for w2 in software_2_split:
            if w1 == w2 and w1 != 'linux':
                # not exactly the same
                return False, software_2
    # not same
    return False, False